import os


#Get every txt file in training_data folder

def load_training_data(folder_path):
    training_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    training_data = {}
    for file_name in training_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            training_data[file_name] = f.read()
    return training_data

if __name__ == '__main__':
    training_folder = 'training_data'
    all_training_data = load_training_data(training_folder)
    for file_name, content in all_training_data.items():
        print(f"--- Content of {file_name} ---")
        print(content)
        print("\n")
