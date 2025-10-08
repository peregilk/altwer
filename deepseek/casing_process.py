import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse

def process_file(input_file, template, output_file, api_key):
    """
    Processes a JSON-lines input file using the OpenAI API and saves the output.

    Args:
        input_file (str): Path to the input JSON-lines file.
        template (str): Template string for generating prompts.
        output_file (str): Path to the output JSON-lines file.
        api_key (str): API key for OpenAI.

    Returns:
        None
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Check for already processed lines
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as output:
            processed_count = sum(1 for _ in output)

    # Count total lines in the input file
    with open(input_file, "r", encoding="utf-8") as input_f:
        total_lines = sum(1 for _ in input_f)

    try:
        with open(input_file, "r", encoding="utf-8") as input_f, \
             open(output_file, "a", encoding="utf-8") as output_f:

            # Skip already processed lines
            for _ in range(processed_count):
                next(input_f)

            # Process remaining lines with a progress bar
            with tqdm(total=total_lines, initial=processed_count, desc="Processing") as progress_bar:
                for line in input_f:
                    # Parse the JSON object
                    data = json.loads(line.strip())

                    # Ensure required fields are present
                    if "verbatim" not in data or "orthographic" not in data:
                        raise ValueError("Each line in the input file must contain 'verbatim' and 'orthographic' fields.")

                    # Fill in the template with the current line's data
                    user_prompt = template.format(
                        verbatim=data["verbatim"],
                        orthographic=data["orthographic"]
                    )

                    # Generate response from the API
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": user_prompt},
                        ],
                        stream=False
                    )

                    # Extract and parse the JSON content from the response
                    content = response.choices[0].message.content.strip()
                    try:
                        if content.startswith("```json") and content.endswith("```"):
                            json_block = content[7:-3].strip()
                        else:
                            json_block = content
                        parsed_json = json.loads(json_block)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse JSON content: {e}")

                    # Add the new fields to the original data
                    data["orthographic-casing"] = parsed_json.get("orthographic-casing", "N/A")
                    data["verbatim-casing"] = parsed_json.get("verbatim-casing", "N/A")

                    # Log a warning if expected fields are missing
                    if "orthographic-casing" not in parsed_json or "verbatim-casing" not in parsed_json:
                        print(f"Warning: Missing expected fields in model output for input: {data}")

                    # Write the updated JSON object to the output file
                    json.dump(data, output_f)
                    output_f.write("\n")  # Ensure newline for JSONLines format

                    # Update progress bar
                    progress_bar.update(1)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_template(template_file):
    """
    Loads the template file.

    Args:
        template_file (str): Path to the template file.

    Returns:
        str: Template content as a string.
    """
    try:
        with open(template_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The template file '{template_file}' was not found.")


def calculate_output_filename(input_file):
    """
    Generates the output filename based on the input filename.

    Args:
        input_file (str): Input filename.

    Returns:
        str: Output filename.
    """
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}_processed.jsonl"


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process a JSON-lines file with OpenAI API.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON-lines file.")
    parser.add_argument("--template_file", default="casing_template.txt", help="Path to the template file (default: casing_template.txt).")
    args = parser.parse_args()

    # Load environment variables
    api_key = os.getenv("DeepSeekApi")
    if not api_key:
        raise EnvironmentError("Environment variable 'DeepSeekApi' is not set.")

    # Load template and calculate output file name
    template_content = load_template(args.template_file)
    output_file_name = calculate_output_filename(args.input_file)

    # Process the file
    process_file(
        input_file=args.input_file,
        template=template_content,
        output_file=output_file_name,
        api_key=api_key
    )
