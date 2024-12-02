# # import os
# # from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# # from peft import LoraConfig, get_peft_model
# # from datasets import Dataset

# # # Step 1: Load training and validation text files
# # def load_text_file(file_path):
# #     with open(file_path, "r", encoding="utf-8") as f:
# #         text = f.read()
# #     return text

# # # Replace with your dataset paths
# # train_file_path = "uploaded_docs/dhoni.txt"
# # valid_file_path = "uploaded_docs/sachin.txt"

# # # Load training and validation data
# # train_text = load_text_file(train_file_path)
# # valid_text = load_text_file(valid_file_path)

# # # Preprocess data into instruction-response pairs or chunks
# # def preprocess_data(text):
# #     return [{"instruction": "Analyze this text:", "response": chunk.strip()} 
# #             for chunk in text.split("\n\n") if chunk.strip()]

# # train_data = preprocess_data(train_text)
# # valid_data = preprocess_data(valid_text)

# # # Create Hugging Face datasets
# # train_dataset = Dataset.from_list(train_data)
# # valid_dataset = Dataset.from_list(valid_data)

# # # Step 2: Tokenize the datasets
# # def tokenize_function(example):
# #     tokens = tokenizer(
# #         example["instruction"],
# #         text_pair=example["response"],
# #         truncation=True,
# #         padding="max_length",
# #         max_length=512,
# #     )
# #     # Add labels to the tokenized data
# #     tokens["labels"] = tokens["input_ids"].copy()
# #     return tokens

# # # Load tokenizer for Llama 3.2 1B model
# # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# # tokenized_train_dataset = train_dataset.map(tokenize_function)
# # tokenized_valid_dataset = valid_dataset.map(tokenize_function)

# # # from langchain_community.llms import Ollama

# # # model_name = Ollama(model="llama3.2")  # Adjust the model name as necessary


# # # Step 3: Load the Llama 3.2 1B model
# # model_name = "NousResearch/Llama-3.2-1B"  # Update to your local Llama 3.2 1B model path
# # model = AutoModelForCausalLM.from_pretrained(model_name)

# # # Step 4: Apply LoRA for fine-tuning
# # lora_config = LoraConfig(
# #     task_type="CAUSAL_LM",
# #     r=16,
# #     lora_alpha=32,
# #     lora_dropout=0.1
# # )
# # model = get_peft_model(model, lora_config)

# # # Step 5: Define training arguments
# # training_args = TrainingArguments(
# #     output_dir="llama3.2",
# #     per_device_train_batch_size=2,
# #     gradient_accumulation_steps=4,
# #     num_train_epochs=10,
# #     learning_rate=2e-5,
# #     eval_strategy="steps",  # Use `eval_strategy` instead of `evaluation_strategy`
# #     eval_steps=100,         # Evaluate every 100 steps
# #     save_steps=500,
# #     save_total_limit=2,
# #     logging_dir="./logs",
# #     logging_steps=1,
# #     fp16=True,  # Enable mixed precision for faster training
# # )

# # # Step 6: Initialize Trainer with training and validation datasets
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=tokenized_train_dataset,
# #     eval_dataset=tokenized_valid_dataset,
# #     tokenizer=tokenizer,
# # )

# # # Step 7: Start fine-tuning
# # trainer.train()

# # # Save the fine-tuned model
# # trainer.save_model("Llama-3_2")
# # tokenizer.save_pretrained(".Llama-3_2t")

# # print("Fine-tuning completed. Model saved to 'Llama-3.2_1b'")

# # # Step 8: Evaluate the fine-tuned model
# # results = trainer.evaluate()
# # print("Validation Results:", results)



# # import os
# # from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# # from peft import LoraConfig, get_peft_model
# # from datasets import Dataset

# # # Step 1: Load training and validation text files
# # def load_text_file(file_path):
# #     with open(file_path, "r", encoding="utf-8") as f:
# #         text = f.read()
# #     return text

# # # Replace with your dataset paths
# # train_file_path = "uploaded_docs/dhoni.txt"
# # valid_file_path = "uploaded_docs/sachin.txt"

# # # Load training and validation data
# # train_text = load_text_file(train_file_path)
# # valid_text = load_text_file(valid_file_path)

# # # Preprocess data into instruction-response pairs or chunks
# # def preprocess_data(text):
# #     return [{"instruction": "Analyze this text:", "response": chunk.strip()} 
# #             for chunk in text.split("\n\n") if chunk.strip()]

# # train_data = preprocess_data(train_text)
# # valid_data = preprocess_data(valid_text)

# # # Create Hugging Face datasets
# # train_dataset = Dataset.from_list(train_data)
# # valid_dataset = Dataset.from_list(valid_data)

# # # Step 2: Tokenize the datasets
# # def tokenize_function(example):
# #     tokens = tokenizer(
# #         example["instruction"],
# #         text_pair=example["response"],
# #         truncation=True,
# #         padding="max_length",
# #         max_length=512,
# #     )
# #     # Add labels to the tokenized data
# #     tokens["labels"] = tokens["input_ids"].copy()
# #     return tokens

# # # Load tokenizer for Llama 3.2 1B model
# # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# # tokenized_train_dataset = train_dataset.map(tokenize_function)
# # tokenized_valid_dataset = valid_dataset.map(tokenize_function)

# # # Step 3: Load the Llama 3.2 1B model
# # model_name = "NousResearch/Llama-3.2-1B"  # Update to your local Llama 3.2 1B model path
# # model = AutoModelForCausalLM.from_pretrained(model_name)

# # # Step 4: Apply LoRA for fine-tuning
# # lora_config = LoraConfig(
# #     task_type="CAUSAL_LM",
# #     r=16,
# #     lora_alpha=32,
# #     lora_dropout=0.1
# # )
# # model = get_peft_model(model, lora_config)

# # # Step 5: Define training arguments
# # training_args = TrainingArguments(
# #     output_dir="llama3.2",
# #     per_device_train_batch_size=2,
# #     gradient_accumulation_steps=4,
# #     num_train_epochs=2,
# #     learning_rate=2e-5,
# #     eval_strategy="steps",  # Use `eval_strategy` instead of `evaluation_strategy`
# #     eval_steps=100,         # Evaluate every 100 steps
# #     save_steps=500,
# #     save_total_limit=2,
# #     logging_dir="./logs",
# #     logging_steps=1,
# #     fp16=True,  # Enable mixed precision for faster training
# # )

# # # Step 6: Initialize Trainer with training and validation datasets
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=tokenized_train_dataset,
# #     eval_dataset=tokenized_valid_dataset,
# #     tokenizer=tokenizer,
# # )

# # # Step 7: Start fine-tuning
# # trainer.train()

# # # Step 8: Save the fine-tuned model and tokenizer in the same directory
# # save_path = "new_llm_model"  # Define the folder name for saving everything

# # # Create the directory if it doesn't exist
# # os.makedirs(save_path, exist_ok=True)

# # # Save the fine-tuned model and tokenizer in the defined folder
# # trainer.save_model(save_path)
# # tokenizer.save_pretrained(save_path)

# # print(f"Fine-tuning completed. Model and tokenizer saved to '{save_path}'")

# # # Step 9: Evaluate the fine-tuned model
# # results = trainer.evaluate()
# # print("Validation Results:", results)

#####################################################################################################################
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset

# # Step 1: Load training and validation text files
# def load_text_file(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()
#     return text

# # Replace with your dataset paths
# train_file_path = "uploaded_docs/dhoni.txt"
# valid_file_path = "uploaded_docs/sachin.txt"

# # Load training and validation data
# train_text = load_text_file(train_file_path)
# valid_text = load_text_file(valid_file_path)

# # Preprocess data into instruction-response pairs or chunks
# def preprocess_data(text):
#     return [{"instruction": "Analyze this text:", "response": chunk.strip()} 
#             for chunk in text.split("\n\n") if chunk.strip()]

# train_data = preprocess_data(train_text)
# valid_data = preprocess_data(valid_text)

# # Create Hugging Face datasets
# train_dataset = Dataset.from_list(train_data)
# valid_dataset = Dataset.from_list(valid_data)

# # Step 2: Tokenize the datasets
# def tokenize_function(example):
#     tokens = tokenizer(
#         example["instruction"],
#         text_pair=example["response"],
#         truncation=True,
#         padding="max_length",  # Ensure padding is applied
#         max_length=512,
#     )
#     # Add labels to the tokenized data
#     tokens["labels"] = tokens["input_ids"].copy()
#     return tokens

# # Load the correct tokenizer for Llama 3.2 model (use the LlamaTokenizer)
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")  # Updated tokenizer for Llama 3.2

# # Set padding token to eos_token (end-of-sequence token)
# tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# # Alternatively, if you want to define a custom pad token:
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Uncomment this line if you want a custom pad token

# tokenized_train_dataset = train_dataset.map(tokenize_function)
# tokenized_valid_dataset = valid_dataset.map(tokenize_function)

# # Step 3: Load the Llama 3.2 model
# model_name = "NousResearch/Llama-3.2-1B"  # Update to your local Llama 3.2 1B model path
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Step 4: Apply LoRA for fine-tuning
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.1
# )
# model = get_peft_model(model, lora_config)

# # Step 5: Define training arguments
# training_args = TrainingArguments(
#     output_dir="llama3.2",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     num_train_epochs=5,
#     learning_rate=2e-5,
#     eval_strategy="steps",  # Use `eval_strategy` instead of `evaluation_strategy`
#     eval_steps=100,         # Evaluate every 100 steps
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=1,
#     fp16=True,  # Enable mixed precision for faster training
# )

# # Step 6: Initialize Trainer with training and validation datasets
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_valid_dataset,
#     tokenizer=tokenizer,
# )

# # Step 7: Start fine-tuning
# trainer.train()

# # Step 8: Save the fine-tuned model and tokenizer in the same directory
# save_path = "new_llama3.2_model"  # Define the folder name for saving everything

# # Create the directory if it doesn't exist
# os.makedirs(save_path, exist_ok=True)

# # Save the fine-tuned model and tokenizer in the defined folder
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)

# print(f"Fine-tuning completed. Model and tokenizer saved to '{save_path}'")

# # Step 9: Evaluate the fine-tuned model
# results = trainer.evaluate()
# print("Validation Results:", results)


##############################################################################################3

# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset

# # Step 1: Create training and validation datasets (instead of loading from files)
# train_text = """Dhoni is a former cricketer and captain of the Indian national team.
# He is known for his cool demeanor and successful career, especially in limited-overs formats."""

# valid_text = """Sachin Tendulkar is a retired cricketer widely regarded as one of the greatest players of all time.
# He holds the record for most runs in both Test and ODI cricket."""

# # Preprocess data into instruction-response pairs or chunks
# def preprocess_data(text):
#     return [{"instruction": "Analyze this text:", "response": chunk.strip()} 
#             for chunk in text.split("\n\n") if chunk.strip()]

# train_data = preprocess_data(train_text)
# valid_data = preprocess_data(valid_text)

# # Create Hugging Face datasets
# train_dataset = Dataset.from_list(train_data)
# valid_dataset = Dataset.from_list(valid_data)

# # Step 2: Tokenize the datasets
# def tokenize_function(example):
#     tokens = tokenizer(
#         example["instruction"],
#         text_pair=example["response"],
#         truncation=True,
#         padding="max_length",  # Ensure padding is applied
#         max_length=512,
#     )
#     # Add labels to the tokenized data
#     tokens["labels"] = tokens["input_ids"].copy()
#     return tokens

# # Load the correct tokenizer for Llama 3.2 model (use the LlamaTokenizer)
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")  # Updated tokenizer for Llama 3.2

# # Set padding token to eos_token (end-of-sequence token)
# tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# # Alternatively, if you want to define a custom pad token:
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Uncomment this line if you want a custom pad token

# tokenized_train_dataset = train_dataset.map(tokenize_function)
# tokenized_valid_dataset = valid_dataset.map(tokenize_function)

# # Step 3: Load the Llama 3.2 model
# model_name = "NousResearch/Llama-3.2-1B"  # Update to your local Llama 3.2 1B model path
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Step 4: Apply LoRA for fine-tuning
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.1
# )
# model = get_peft_model(model, lora_config)

# # Step 5: Define training arguments
# training_args = TrainingArguments(
#     output_dir="llama3.2_v1",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     num_train_epochs=10,
#     learning_rate=2e-5,
#     eval_strategy="steps",  # Use `eval_strategy` instead of `evaluation_strategy`
#     eval_steps=100,         # Evaluate every 100 steps
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=1,
#     fp16=True,  # Enable mixed precision for faster training
# )

# # Step 6: Initialize Trainer with training and validation datasets
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_valid_dataset,
#     tokenizer=tokenizer,
# )

# # Step 7: Start fine-tuning
# trainer.train()

# # Step 8: Save the fine-tuned model and tokenizer in the same directory
# save_path = "new_llama3.2_model_v1"  # Define the folder name for saving everything

# # Create the directory if it doesn't exist
# os.makedirs(save_path, exist_ok=True)

# # Save the fine-tuned model and tokenizer in the defined folder
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)

# print(f"Fine-tuning completed. Model and tokenizer saved to '{save_path}'")

# # Step 9: Evaluate the fine-tuned model
# results = trainer.evaluate()
# print("Validation Results:", results)


##########################################################################################################

# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# from datasets import Dataset

# # Step 1: Preprocess the training and validation datasets
# train_text = """Dhoni is a former cricketer and captain of the Indian national team.
# He is known for his cool demeanor and successful career, especially in limited-overs formats."""

# valid_text = """Sachin Tendulkar is a retired cricketer widely regarded as one of the greatest players of all time.
# He holds the record for most runs in both Test and ODI cricket."""


# # Preprocess the data
# def preprocess_data(text):
#     return [{"instruction": "Explain:", "response": chunk.strip()} 
#             for chunk in text.split("\n") if chunk.strip()]

# train_data = preprocess_data(train_text)
# valid_data = preprocess_data(valid_text)

# # Create Hugging Face datasets
# train_dataset = Dataset.from_list(train_data)
# valid_dataset = Dataset.from_list(valid_data)

# # Step 2: Load the tokenizer and set padding token
# model_name = "NousResearch/Llama-3.2-1B"  # Replace with the correct path to your local model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token



# def tokenize_function(example):
#     tokens = tokenizer(
#         example["instruction"],
#         text_pair=example["response"],
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )
#     tokens["labels"] = tokens["input_ids"].copy()  # Copy input_ids to labels
#     return tokens

# tokenized_train_dataset = train_dataset.map(tokenize_function)
# tokenized_valid_dataset = valid_dataset.map(tokenize_function)

# # Step 3: Load the Llama model and apply LoRA for fine-tuning
# model = AutoModelForCausalLM.from_pretrained(model_name)

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.1
# )
# model = get_peft_model(model, lora_config)

# # Step 4: Define training arguments
# training_args = TrainingArguments(
#     output_dir="llama3.2_v3",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     num_train_epochs=3,  # Start with fewer epochs
#     learning_rate=1e-4,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_steps=100,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,
#     optim="adamw_torch",
# )

# # Step 5: Initialize Trainer and start training
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_valid_dataset,
#     tokenizer=tokenizer,
# )

# trainer.train()

# # Save the fine-tuned model and tokenizer
# save_path = "new_llama3.2_model_v3"
# os.makedirs(save_path, exist_ok=True)
# trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)

# print(f"Fine-tuning completed. Model and tokenizer saved to '{save_path}'")

##################################################################################33

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Step 1: Preprocess the training and validation datasets
# train_text = """Sk aka Saravanabramman is a dancer, actor and AI engineer. he lives in coimbatore.
#  He like to dance, act and travel. he is 29 years old. he is married. He was studied in govt school.
#    he did UG in "Robotics" at PSG college of technology,
#    and done his PG in "Machine learning" at Amrita university. he love his family. 
#    He is having 7 years of work experience. He love his job"""

# valid_text = """sk is born in coimbatore, he is a dance and singer."""

train_text = """
Sk aka Saravanabramman is a dancer, actor, and AI engineer. He lives in Coimbatore, a city in Tamil Nadu.
He enjoys dancing, acting, and traveling around the world. Sk is 29 years old and married.
He studied in a government school before pursuing his higher education.
He did his undergraduate degree in "Robotics" at PSG College of Technology.
After that, he completed his postgraduate studies in "Machine Learning" at Amrita University.
He loves his family very much and is very close to them.
Sk has over 7 years of experience working in AI and machine learning.
He loves his job and enjoys solving complex problems with technology.
Sk is passionate about using AI to make a positive impact in the world.
He believes that technology can help improve lives and make the world a better place.
Sk also loves learning new things and is always excited to explore new technologies.
In his free time, Sk enjoys watching movies, reading books, and spending time with his family.
He is a huge fan of science fiction and is always interested in the latest advancements in technology.
Sk often attends conferences and workshops to stay updated on the latest trends in AI and machine learning.
He is also an advocate for the responsible use of AI in society.
Sk works as an AI engineer in a well-known tech company in Coimbatore.
He is always on the lookout for new ways to innovate and bring positive change through AI.
Sk is a big fan of fitness and makes sure to stay active and healthy.
He practices yoga every morning and loves running in the evenings.
Sk enjoys traveling to new places and experiencing different cultures.
He believes that traveling broadens one's perspective and helps in personal growth.
Sk has visited many countries and looks forward to visiting more in the future.
He has a keen interest in the intersection of AI and healthcare.
Sk hopes to use his skills in AI to create technologies that can help improve the healthcare system.
He loves solving real-world problems and enjoys the challenge of finding innovative solutions.
Sk's role as an AI engineer involves working with large datasets and building machine learning models.
He believes that AI can revolutionize industries like healthcare, education, and finance.
Sk is a creative thinker and enjoys brainstorming ideas with his colleagues.
He is known for his problem-solving skills and his ability to think outside the box.
Sk is also an advocate for women in tech and supports diversity in the tech industry.
He believes that diversity leads to better innovation and more inclusive solutions.
Sk's goal is to become a thought leader in the AI community and inspire others to follow their passion.
Sk is a lifelong learner and believes that learning never stops.
He enjoys reading research papers and staying updated on the latest advancements in AI.
Sk is an active member of various AI and tech communities online.
He believes in the power of collaboration and enjoys working with teams to solve complex problems.
He is always looking for new opportunities to learn and grow in his career.
Sk is a firm believer in the importance of ethical AI and ensuring that AI technologies are used for good.
He believes that AI can help solve some of the world's biggest challenges, from climate change to disease prevention.
Sk is a mentor to several junior engineers and enjoys helping others grow in their careers.
He believes that mentorship is an important part of personal and professional development.
Sk is a strong advocate for work-life balance and makes sure to prioritize his family and personal time.
He believes that taking care of oneself is key to being productive and successful in both work and life.
Sk's journey in AI started with a fascination for robotics as a child.
He has always been curious about how things work and how technology can be used to improve lives.
Sk hopes to make a lasting impact on the world through his work in AI and machine learning.
He dreams of one day leading a team of AI engineers and making groundbreaking contributions to the field.
Sk is inspired by the work of other AI pioneers and hopes to be a source of inspiration for others.
He believes that anyone can learn AI if they have the passion and dedication to do so.
Sk encourages young people to pursue careers in STEM and follow their dreams.
He believes that AI is the future, and he is excited to be a part of shaping that future.
Sk is always looking for new projects and challenges to work on.
He enjoys collaborating with like-minded individuals and teams to bring ideas to life.
Sk believes that innovation thrives when people work together and share ideas.
He is constantly seeking new ways to use AI to make a positive impact on society.
Sk is passionate about education and believes that everyone should have access to quality education.
He hopes to use AI to create educational tools that can help people learn more effectively.
Sk enjoys giving back to the community and participates in various charity events and causes.
He is passionate about helping others and believes in making the world a better place.
Sk has worked on several AI projects related to healthcare, including AI for diagnosing diseases.
He believes that AI can help doctors and healthcare providers deliver better care to patients.
Sk's work in AI has earned him recognition from his peers and industry leaders.
He is proud of the work he has done and is excited about the future of AI.
Sk is always ready to take on new challenges and is excited about the possibilities of AI.
He believes that AI will continue to evolve and become an even more integral part of our lives.
Sk has a deep understanding of machine learning algorithms and their applications in real-world scenarios.
He loves experimenting with different machine learning models and learning from the results.
Sk is a huge fan of data science and is always eager to dive into datasets to uncover valuable insights.
He believes that data is one of the most important resources we have today and can be used to make informed decisions.
Sk is passionate about using AI to create smarter cities and more efficient public systems.
He is always thinking about how AI can be applied to solve urban problems and improve the quality of life in cities.
Sk has a positive outlook on the future of AI and believes that it has the potential to revolutionize industries across the board.
He is excited to be a part of the AI revolution and is dedicated to making a difference in the field.
Sk believes in the importance of building strong relationships and collaborating with others to achieve great things.
He values trust, communication, and teamwork and believes these are the key ingredients for success.
"""

valid_text = """
Sk was born in Coimbatore and is known for his work in dance and AI.
He has a passion for both the arts and technology, which makes him unique in his field.
Sk has worked as a dance instructor, actor, and AI engineer.
He has received recognition for his contributions to both the dance and tech industries.
Sk loves working on challenging AI problems and creating innovative solutions.
He is also a talented singer and enjoys performing on stage.
Sk's ability to balance his creative and technical pursuits has earned him a lot of respect.
"""


# Preprocess the data: Split text into individual lines and create dataset with 'text' field
def preprocess_data(text):
    return [{"text": chunk.strip()} for chunk in text.split("\n") if chunk.strip()]

train_data = preprocess_data(train_text)
valid_data = preprocess_data(valid_text)

# Create Hugging Face datasets
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

# Step 2: Load the tokenizer and set padding token
model_name = "NousResearch/Llama-3.2-1B"  # Replace with the correct path to your local model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token

# Tokenize datasets
def tokenize_function(example):
    tokens = tokenizer(
        text=example["text"],  # Use the 'text' field directly
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Set labels to input_ids for causal language modeling
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Step 3: Load the Llama model and apply LoRA for fine-tuning
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type="SEQ2SEQ_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.2
)

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=4,
#     lora_alpha=8,
#     lora_dropout=0.3
# )
# model = get_peft_model(model, lora_config)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="llama3.2_v2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,  # Start with fewer epochs
    learning_rate=1e-4,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=50,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="adamw_torch",
)

# Step 5: Initialize Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model and tokenizer
save_path = "new_llama3.2_model_v2"
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"Fine-tuning completed. Model and tokenizer saved to '{save_path}'")
