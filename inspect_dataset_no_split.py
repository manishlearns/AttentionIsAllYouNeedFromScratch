from datasets import load_dataset

ds = load_dataset('opus_books', 'en-it')
print(f"Type of ds: {type(ds)}")
print(f"Keys in ds: {list(ds.keys())}")
for item in ds:
    print(f"Item in loop: {item}")
    break
