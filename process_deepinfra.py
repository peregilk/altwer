#!/usr/bin/env python3
import os
import json
import argparse
import logging
import re
import threading
import time
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

# Global cost tracking
total_cost = 0.0
cost_lock = threading.Lock()

# Thread-local storage for the API client.
_thread_local = threading.local()

def get_client(api_key):
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
    return _thread_local.client

def load_template(tpath):
    with open(tpath, "r", encoding="utf-8") as f:
        return f.read()

def parse_response(content):
    """
    Attempt to extract a JSON object from the first '{' to the last '}'.
    """
    content = content.strip()
    # Remove any <think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # Remove markdown fences if present
    if content.startswith("```"):
        content = content.strip("`").strip()
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except Exception as e:
            logging.error(f"JSON load error: {e}")
            return None
    return None

def process_line(line, template, api_key, model):
    """
    Processes one input JSON line (with key "text").
    Uses up to 10 retries if the API call fails.
    Returns a JSON string with keys:
      "text" - always the original input text,
      "altwer_reasoning", "altwer_syntax" as parsed from the API response,
      "result" - "OK" if both altwer fields were non-empty; otherwise an error message.
    """
    global total_cost
    try:
        record = json.loads(line.strip())
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {line.strip()} | {e}")
        return None

    input_text = record.get("text", "").strip()
    if not input_text:
        logging.warning("No 'text' field found in record.")
        return None

    # Replace placeholder {text} in template
    prompt = template.replace("{text}", input_text)
    client = get_client(api_key)

    last_error = ""
    response = None
    for attempt in range(10):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=2048,
            )
            break
        except Exception as e:
            last_error = str(e)
            logging.warning(f"Attempt {attempt+1}/10 failed for text: {input_text} | {e}")
            time.sleep(1)
    else:
        # All attempts failed
        output = {
            "text": input_text,
            "altwer_reasoning": "",
            "altwer_syntax": "",
            "result": f"APIError: {last_error}"
        }
        return json.dumps(output, ensure_ascii=False)

    # If using non-stream mode, grab the content
    content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    
    # Accumulate cost if available
    cost_val = getattr(response.usage, "estimated_cost", 0.0) if hasattr(response, "usage") else 0.0
    with cost_lock:
        total_cost += cost_val

    parsed = parse_response(content)
    if not parsed:
        result_status = "ParsingError: Unable to parse response JSON."
        parsed = {}
    else:
        missing = []
        if not parsed.get("altwer_reasoning", "").strip():
            missing.append("altwer_reasoning")
        if not parsed.get("altwer_syntax", "").strip():
            missing.append("altwer_syntax")
        if missing:
            result_status = "MissingFields: " + ", ".join(missing)
        else:
            result_status = "OK"

    output = {
        "text": input_text,
        "altwer_reasoning": parsed.get("altwer_reasoning", ""),
        "altwer_syntax": parsed.get("altwer_syntax", ""),
        "result": result_status
    }
    return json.dumps(output, ensure_ascii=False)

def process_file_parallel(input_file, template, output_file, api_key, model, num_workers, max_items=None):
    """
    Reads all lines from input_file.
    Skips lines already in output_file (by count).
    Processes up to max_items new lines (if set) and appends them to output_file.
    Returns the number of new lines processed.
    """
    out_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_out:
            out_count = sum(1 for _ in f_out)

    with open(input_file, "r", encoding="utf-8") as f_in:
        in_lines = f_in.readlines()
    total_in = len(in_lines)

    remain = total_in - out_count
    if remain <= 0:
        logging.info("No new lines to process (output already complete).")
        return 0

    to_do = remain
    if max_items is not None:
        to_do = min(to_do, max(0, max_items - out_count))
    if to_do <= 0:
        logging.info("Reached max_items threshold; no new lines processed.")
        return 0

    lines_to_process = in_lines[out_count:out_count + to_do]
    logging.info(f"Processing {len(lines_to_process)} new lines (from {out_count} to {out_count + to_do}).")

    with ThreadPoolExecutor(max_workers=num_workers) as executor, \
         open(output_file, "a", encoding="utf-8") as out_f:
        for processed in tqdm(
            executor.map(process_line, lines_to_process, repeat(template), repeat(api_key), repeat(model)),
            total=len(lines_to_process),
            desc="Processing new lines"
        ):
            if processed:
                out_f.write(processed + "\n")
                out_f.flush()
    return len(lines_to_process)

def repair_file(input_file, output_file, template, api_key, model, num_workers):
    """
    Reads both input_file and output_file line by line.
    For each line (up to the minimum number of lines in both files),
    if the parsed output record's "result" field is not "OK", reprocess that line from input.
    Overwrites output_file with the fixed records.
    """
    if not os.path.exists(output_file):
        logging.warning("No output file found; nothing to repair.")
        return

    with open(input_file, "r", encoding="utf-8") as f_in:
        in_lines = f_in.readlines()
    with open(output_file, "r", encoding="utf-8") as f_out:
        out_lines = f_out.readlines()

    min_len = min(len(in_lines), len(out_lines))
    fixed_output = [None] * min_len
    reproc_indices = []

    for idx in range(min_len):
        line_out = out_lines[idx].strip()
        try:
            data = json.loads(line_out)
            if data.get("result", "") != "OK":
                reproc_indices.append(idx)
            else:
                fixed_output[idx] = line_out
        except json.JSONDecodeError:
            reproc_indices.append(idx)

    if reproc_indices:
        logging.info(f"Repairing {len(reproc_indices)} lines out of {min_len}.")
    else:
        logging.info("No lines to repair.")
        return

    reproc_input = [in_lines[i] for i in reproc_indices]
    results_map = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {executor.submit(process_line, reproc_input[i], template, api_key, model): reproc_indices[i]
                      for i in range(len(reproc_indices))}
        for fut in tqdm(future_map, total=len(reproc_indices), desc="Repairing lines"):
            idx = future_map[fut]
            try:
                res = fut.result()
                if res:
                    results_map[idx] = res.strip()
                else:
                    results_map[idx] = out_lines[idx].strip()
            except Exception as e:
                logging.error(f"Repair error at line {idx}: {e}")
                results_map[idx] = out_lines[idx].strip()

    for idx in reproc_indices:
        fixed_output[idx] = results_map.get(idx, out_lines[idx].strip())

    # Append any extra lines from the output if present.
    if len(out_lines) > min_len:
        fixed_output.extend(line.strip() for line in out_lines[min_len:])

    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in fixed_output:
            f_out.write(line + "\n")
    logging.info(f"Repaired {len(reproc_indices)} lines.")

def main():
    parser = argparse.ArgumentParser(description="Process partial lines and auto-repair entries where result != OK.")
    parser.add_argument("--input_file", required=True, help="Input JSONL file with key 'text'.")
    parser.add_argument("--template_file", required=True, help="Prompt template file.")
    parser.add_argument("--output_file", required=True, help="Output JSONL file.")
    parser.add_argument("--processes", type=int, default=50, help="Number of parallel workers.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1", help="Model to use.")
    parser.add_argument("--max_items", type=int, default=None, help="Stop normal processing once output lines >= this.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    api_key = os.getenv("DEEP_INFRA")
    if not api_key:
        raise EnvironmentError("DEEP_INFRA environment variable not set.")

    # Ensure output directory exists.
    odir = os.path.dirname(args.output_file)
    if odir:
        os.makedirs(odir, exist_ok=True)

    template_content = load_template(args.template_file)

    # Count lines in output and input.
    out_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f_out:
            out_count = sum(1 for _ in f_out)
    with open(args.input_file, "r", encoding="utf-8") as f_in:
        total_in = sum(1 for _ in f_in)

    logging.info(f"Output file has {out_count} lines; input file has {total_in} lines.")

    # Decide on processing mode:
    # If output is incomplete (< total_in and, if max_items is set, < max_items), process new lines.
    # Otherwise, enter repair mode.
    if out_count < total_in and (args.max_items is None or out_count < args.max_items):
        processed_now = process_file_parallel(
            input_file=args.input_file,
            template=template_content,
            output_file=args.output_file,
            api_key=api_key,
            model=args.model,
            num_workers=args.processes,
            max_items=args.max_items
        )
        if processed_now:
            logging.info(f"Processed {processed_now} new lines.")
    else:
        logging.info("No new lines to process; entering repair mode.")
        repair_file(
            input_file=args.input_file,
            output_file=args.output_file,
            template=template_content,
            api_key=api_key,
            model=args.model,
            num_workers=args.processes
        )

    print(f"Total estimated cost: {total_cost:.8f}")

if __name__ == "__main__":
    main()