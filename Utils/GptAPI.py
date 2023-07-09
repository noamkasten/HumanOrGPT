import pandas as pd
import openai
import os


# Load your Excel file
df = pd.read_excel('/content/Rephrased_Our_testset.xlsx')

sentences = df['Human Review'].tolist()

print(len(sentences))

PROMPT = "Please rephrase the following restaurant reviews in your own words line by line independently. Do not number the rephrased sentences, Do not leave empty lines, And do not add bullet points, {}:\n{}"
SHAP_EDITION = """Instructions: To sound more human and less like a bot,
try to keep the specific food words like: bacon, bread, steak, hamburger, meat, pizza, noodles.
Also try to use more words about the flavor such as spicy. From time to time, Use more exclamation marks.
Try to avoid using too much "the" and "this" and general generic words like: service, place, establishment. Also try to avoid using a periods and commas in the sentences.
And here are the sentences to rephrase
"""


current_prompt=""
# Set your OpenAI API key
openai.api_key = '' # REMOVE THIS BEFORE UPLOADING TO GITLAB


merge_20_sentences = []
rephrased_sentences = list()
rephrase = ""


last_worked = -1
# Open the file in read mode

file_path = "/content/output.txt"
# Check if the file exists
if os.path.exists(file_path):
    # Open the file in read mode
    with open(file_path, "r") as file:
        # Read the contents of the file
        contents = file.readlines()
    # Extract the number from the contents
    last_worked = int(contents[-1])
    if not last_worked:
      last_worked = -1
    # Use the retrieved number as needed
else:
    last_worked = -1

print("++++++ Starting retrieving from idx: {}".format(last_worked+1))

# Check if the file exists
if os.path.exists("/content/ready_so_far.txt"):
    # Open and read the file
    with open("/content/ready_so_far.txt", "r") as file:
        # Read all lines from the file and convert them to a list
        my_list = file.readlines()

    # Strip newline characters from each line in the list
    rephrased_sentences = [line.strip() for line in my_list]
else:
    rephrased_sentences = list()

for idx, sentence in enumerate(sentences, start=last_worked+1):
    merge_20_sentences.append(sentence)
    if (idx+1) % 20 == 0 or (idx+1) == len(sentences):
        print(idx+1)
        sentences_str = '\n'.join(merge_20_sentences)
        current_prompt = PROMPT.format(SHAP_EDITION,sentences_str)
        # print(current_prompt)
        merge_20_sentences = []
        messages = [{"role": "system", "content": current_prompt}]

        diff_from_last_run = idx - last_worked
        length_of_returned = 0

        while diff_from_last_run != length_of_returned:
          # Get a rephrase from the AI model
          chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

          rephrase = chat.choices[0].message.content
          print(rephrase)
          sentences_returned = rephrase.split("\n")
          length_of_returned = len(sentences_returned)
          if diff_from_last_run != length_of_returned:
            print("!!!!!!!!!!! MISMATCH, trying again !!!!!!!!!!!")
        rephrased_sentences += sentences_returned
        last_successful = "last successful idx:\n{}".format(idx)
        last_worked = idx

        with open("/content/ready_so_far.txt", "a") as file:
          # Write each element of the list on a separate line
          for item in sentences_returned:
              file.write(str(item) + "\n")

        with open("/content/output.txt", "w") as file:
            # Write the string to the file
            file.write(last_successful)
