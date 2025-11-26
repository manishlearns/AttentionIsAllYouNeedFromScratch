from datasets import load_dataset

ds = load_dataset('opus_books', 'en-it', split='train')
print("Dataset loaded.")
print(f"First item: {ds[0]}")
print(f"Type of first item: {type(ds[0])}")
print(f"Type of translation: {type(ds[0]['translation'])}")
