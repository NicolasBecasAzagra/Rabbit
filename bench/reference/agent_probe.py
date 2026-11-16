#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Probe whether a checkpoint closes the conversation of the agent.

The agent of this repository asks the model for one tool call per step. The
135 million parameter checkpoint reads a file and then asks for the same file
again, so the loop stops on the repetition and never reaches `done`. Before a
bigger checkpoint costs hours of emulation, this script answers the question on
the host, in seconds: with the same prompt, does the model call `done`?

The prompt is built the way `rabbit-agent` builds it. The system text and the
default task are read from `rabbit-agent/src/main.rs`, so the two programs
cannot drift apart, and the tool results are fixtures with the shape the tools
return.

The grammar of the agent forces every answer to be a well formed call of a
declared tool. The probe imitates the part that decides the loop: it writes
`{"tool":"` itself, picks the declared tool the model finds most likely, and
lets the model write the arguments. Choosing the tool is what the model has
to get right, so a model that never picks `done` here will not pick it in the
agent either. The arguments are not constrained, and the run of the agent is
the proof.

Usage:
    python3 bench/reference/agent_probe.py --model-dir DIR [--quant 0|8|4]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_MAIN = ROOT / "rabbit-agent" / "src" / "main.rs"
AGENT_TOOLS = ROOT / "rabbit-agent" / "src" / "tools.rs"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# The values the loop uses: MAX_STEPS and MAX_NEW_TOKENS of rabbit-agent.
MAX_STEPS = 4
MAX_NEW_TOKENS = 96

# What the tools answer, in the shape of `tools.rs`. The server of the rig
# answers a short text, so the answer of the task is in the first result.
FIXTURES = {
    "http_get": (
        "status=200 reason=OK bytes=14 headers=content-type=text/plain "
        "content=hello rabbit retries=0 rt_us=1200"
    ),
    "read_file": "bytes=14 read=14 content=hello rabbit",
    "list_dir": "error ENOSYS",
    "write_file": "error EROFS",
    "compute": "value=4",
    "net_send": "bytes=5 content=hello",
}


def rust_string(source: str, name: str) -> str:
    """Read a `pub const NAME: &str = "...";` of the agent and unescape it."""
    match = re.search(
        r"pub const " + name + r": &str = \"((?:[^\"\\]|\\.)*)\";", source
    )
    if not match:
        sys.exit(f"{name} not found in {AGENT_MAIN}")
    return json.loads('"' + match.group(1) + '"')


def tool_names(source: str) -> list:
    """The names of the registry, in the order the prompt lists them."""
    match = re.search(r"pub const NAMES: \[&str; \d+\] = \[(.*?)\];", source, re.S)
    if not match:
        sys.exit(f"NAMES not found in {AGENT_TOOLS}")
    return re.findall(r'"([a-z_]+)"', match.group(1))


def block(role: str, text: str) -> str:
    return f"{IM_START}{role}\n{text}\n{IM_END}\n"


def tool_prompt(system: str, task: str, tools: list, history: list) -> str:
    """The same assembly as `chat::tool_prompt`."""
    prompt = block("system", f"{system}\nAvailable tools: {', '.join(tools)}")
    for role, text in history:
        prompt += block(role, text)
    prompt += block("user", task)
    return prompt + f"{IM_START}assistant\n"


def quantize_linear(model, bits: int) -> int:
    """Round every linear weight to blocks of 32 with an fp16 scale.

    The scale of a block is its largest magnitude over the largest integer of
    the format, 127 for eight bits and 7 for four. It is an approximation of
    what `rabbit-infer-quantize` writes, good enough to see whether the answer
    of the model survives the rounding.
    """
    import torch

    top = 127 if bits == 8 else 7
    count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.data
        if weight.shape[-1] % 32:
            continue
        blocks = weight.reshape(-1, 32)
        scale = (blocks.abs().amax(dim=1, keepdim=True) / top).to(torch.float16)
        scale = scale.to(weight.dtype).clamp(min=1e-8)
        rounded = torch.clamp(torch.round(blocks / scale), -top, top) * scale
        module.weight.data = rounded.reshape(weight.shape)
        count += 1
    return count


PREFIX = '{"tool":"'


def choose_tool(model, tokenizer, prompt: str, tools: list):
    """Pick the declared tool with the highest log probability after PREFIX."""
    import torch

    base = len(prompt + PREFIX)
    scores = {}
    for name in tools:
        text = prompt + PREFIX + name + '","args":'
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        ids = encoded.input_ids
        offsets = encoded.offset_mapping[0]
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for position in range(1, ids.shape[1]):
            if int(offsets[position][0]) >= base:
                total += float(logprobs[position - 1, ids[0, position]])
        scores[name] = total
    best = max(scores, key=scores.get)
    return best, scores


def write_args(model, tokenizer, prompt: str, tool: str, end_ids: list) -> str:
    """Let the model write the arguments of the chosen tool, up to the object."""
    import torch

    text = prompt + PREFIX + tool + '","args":'
    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=end_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    written = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    try:
        value, end = json.JSONDecoder().raw_decode(written.strip())
        return json.dumps(value, sort_keys=True) if isinstance(value, dict) else "{}"
    except json.JSONDecodeError:
        return "{}"


# The variants of the prompt under test: text appended to the system block, and
# whether the result of a tool is framed with a hint to finish.
EXAMPLE = (
    ' Example: for the task "Read /a" you call {"tool":"read_file","args":'
    '{"path":"/a"}}, the result is "bytes=2 content=hi", and then you call '
    '{"tool":"done","args":{"text":"hi"}}.'
)
VARIANTS = {
    "none": ("", False),
    "system": (
        " When a tool has returned what the task asks for, call done with it."
        " Never call the same tool twice.",
        False,
    ),
    "frame": ("", True),
    "both": (
        " When a tool has returned what the task asks for, call done with it."
        " Never call the same tool twice.",
        True,
    ),
    "example": (EXAMPLE, False),
    "example-frame": (EXAMPLE, True),
}


def run_variant(model, tokenizer, system, task, tools, end_ids, frame):
    """Run the loop of the agent once and say how it ended."""
    history = []
    previous = None
    steps = []
    outcome = "ran out of steps"

    for step in range(1, MAX_STEPS + 1):
        prompt = tool_prompt(system, task, tools, history)
        tool, scores = choose_tool(model, tokenizer, prompt, tools)
        call_args = write_args(model, tokenizer, prompt, tool, end_ids)
        written = f'{PREFIX}{tool}","args":{call_args}}}'
        ranked = sorted(scores, key=scores.get, reverse=True)
        gap = scores[ranked[0]] - scores["done"]
        steps.append(
            {"step": step, "tool": tool, "args": call_args, "top3": ranked[:3],
             "done_gap": round(gap, 2)}
        )
        print(
            f"  step {step}: {tool} {call_args[:80]}  "
            f"(top3: {', '.join(ranked[:3])}; done is {gap:.2f} nats behind)"
        )

        request = f"{tool} {call_args}"
        if request == previous:
            outcome = "repeated the call"
            break
        previous = request
        if tool == "done":
            outcome = "done"
            break
        result = FIXTURES.get(tool, "error unknown tool")
        if frame:
            result = (
                f"Result of {tool}: {result}\n"
                "If this answers the task, call done with the answer."
            )
        history.append(("assistant", written))
        history.append(("user", result))
    return outcome, steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--quant", type=int, default=0, choices=(0, 4, 8))
    parser.add_argument("--endpoint", default="http://10.0.2.2:8080/")
    parser.add_argument(
        "--variants",
        default="none",
        help="comma separated: " + ", ".join(VARIANTS),
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    names = args.variants.split(",")
    for name in names:
        if name not in VARIANTS:
            sys.exit(f"unknown variant {name}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = AGENT_MAIN.read_text()
    system = rust_string(source, "SYSTEM")
    task = rust_string(source, "DEFAULT_TASK").replace("{endpoint}", args.endpoint)
    tools = tool_names(AGENT_TOOLS.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32
    )
    model.eval()
    if args.quant:
        rounded = quantize_linear(model, args.quant)
        print(f"quantized {rounded} linear layers to {args.quant} bits")

    end_ids = [tokenizer.convert_tokens_to_ids(IM_END), tokenizer.eos_token_id]
    runs = []
    for name in names:
        system_text, frame = VARIANTS[name]
        outcome, steps = run_variant(
            model, tokenizer, system + system_text, task, tools, end_ids, frame
        )
        print(f"variant {name}: {outcome}")
        runs.append({"variant": name, "outcome": outcome, "steps": steps})

    if args.out:
        report = {"model_dir": args.model_dir, "quant": args.quant, "runs": runs}
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(run["outcome"] == "done" for run in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
