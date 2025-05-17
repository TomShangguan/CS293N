import os
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = ""

# Configure the Gemini API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_description(prompt_text):
    """Generate a description using the Gemini model."""
    try:
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return ""

def process_folder(folder_path, label, output_file):
    """Process all .txt files in the specified folder, generate descriptions, and write to the output file."""
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    packet_headers = f.read()

                # Construct the prompt
                prompt = (
                    f"The following are multiple packet header information of a network state:\n{packet_headers}\n"
                    "Please generate a concise description summarizing the key characteristics of this network state."
                )

                # Generate the description
                description = generate_description(prompt)

                # Write to the output file
                out_f.write(f"File: {filename}\n")
                out_f.write(f"Label: {'Attack' if label == '0' else 'Benign'}\n")
                out_f.write(f"Description: {description}\n")
                out_f.write("=" * 50 + "\n")

def generate_state_descriptions():
    base_path = "attack_data"
    output_attack = "attack_descriptions.txt"
    output_benign = "benign_descriptions.txt"

    # Process attack data
    attack_folder = os.path.join(base_path, "0")
    process_folder(attack_folder, label='0', output_file=output_attack)

    # Process benign data
    benign_folder = os.path.join(base_path, "1")
    process_folder(benign_folder, label='1', output_file=output_benign)
