import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a JSON-lines file with OpenAI API.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON-lines file.")
    parser.add_argument("--template_file", default="casing_template.txt", help="Path to the template file (default: casing_template.txt).")
    args = parser.parse_args()

    input_file_path = args.input_file
    template_file_path = args.template_file
    output_file_path = f"{input_file_path}_processed.jsonl"

    # Read API key from environment variable
    api_key = os.getenv("DeepSeekApi")
    if not api_key:
        raise EnvironmentError("Environment variable 'DeepSeekApi' is not set.")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Read the template
    try:
        with open(template_file_path, "r", encoding="utf-8") as template_file:
            template = template_file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The template file '{template_file_path}' was not found.")

    # Check for already processed lines
    processed_count = 0
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as output_file:
            processed_count = sum(1 for _ in output_file)

    # Count total lines in the input file
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        total_lines = sum(1 for _ in input_file)

    try:
        with open(input_file_path, "r", encoding="utf-8") as input_file, \
             open(output_file_path, "a", encoding="utf-8") as output_file:

            # Skip already processed lines
            for _ in range(processed_count):
                next(input_file)

            # Process remaining lines with a progress bar
            with tqdm(total=total_lines, initial=processed_count, desc="Processing") as progress_bar:
                for line in input_file:
                    # Parse the JSON object
                    data = json.loads(line.strip())

                    # Ensure the JSON object contains the required fields
                    if "verbatim" not in data or "orthographic" not in data:
                        raise ValueError("Each line in the input file must contain 'verbatim' and 'orthographic' fields.")

                    # Fill in the template with the current line's data
                    user_prompt = template.format(
                        verbatim=data["verbatim"],
                        orthographic=data["orthographic"]
                    )

                    # Generate response
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": user_prompt},
                        ],
                        stream=False
                    )

                    # Extract the JSON content from the assistant's message
                    content = response.choices[0].message.content.strip()

                    # Parse the JSON block inside the content
                    try:
                        if content.startswith("```json") and content.endswith("```"):
                            json_block = content[7:-3].strip()
                        else:
                            json_block = content

                        parsed_json = json.loads(json_block)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse JSON content: {e}")

                    # Safely extract new fields
                    data["orthographic-casing"] = parsed_json.get("orthographic-casing", "N/A")
                    data["verbatim-casing"] = parsed_json.get("verbatim-casing", "N/A")

                    # Log a warning if expected fields are missing
                    if "orthographic-casing" not in parsed_json or "verbatim-casing" not in parsed_json:
                        print(f"Warning: Missing expected fields in model output for input: {data}")

                    # Write the updated JSON object to the output file
                    json.dump(data, output_file)
                    output_file.write("\n")  # Ensure newline for JSONLines format

                    # Update progress bar
                    progress_bar.update(1)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
