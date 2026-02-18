# DE300 HW2 – Movie Recommendation System

## Overview
This project builds a movie recommender using BERT-based movie embeddings (offline) and cosine similarity (online).
Each task is implemented as a function and can be run independently via a CLI entry point.

All S3 inputs/outputs are stored under the `hw2/` prefix (e.g., `hw2/movielens-1m.zip`).

See expected_output.pdf for photos of expected terminal output for each task. 
See fingerson-winter26/hw2 in S3 for expected output files. 

---

## Requirements
Install:
```bash
pip install -r requirements.txt
```

---

## Running the Project

The pipeline is controlled via command-line arguments:

```bash
python de300_hw2.py --tasknum N --bucketname YOUR_BUCKET
```

To run the whole assignment in my bucket:
```bash
python de300_hw2.py --tasknum 1 --bucketname fingerson-winter26
python de300_hw2.py --tasknum 2 --bucketname fingerson-winter26
python de300_hw2.py --tasknum 3 --bucketname fingerson-winter26
python de300_hw2.py --tasknum 4 --bucketname fingerson-winter26
python de300_hw2.py --tasknum 5 --bucketname fingerson-winter26
```

The script supports the following input parameters (default values shown)
- tasknum 1
- bucketname fingerson-winter26
- task2_subset_fraction 0.3
- task2_embeddings_name hw2/movie_subset_embeddings_with_ids.pt
- task4_embeddings_name hw2/FULL_movie_subset_embeddings_with_ids.pt
- task5_profile_name hw2/livs_profile.csv

For example:
```bash
python de300_hw2.py \
  --tasknum 2 \
  --bucketname other-bucket \
  --task2_subset_fraction 0.2 \
  --task2_embeddings_name hw2/my_subset_embeddings.pt
```

## Tasks and Functions

### Task 1 – Download MovieLens Dataset

**Function:**  
`T1_download_movielens(bucket_name)`

**Description:**  
- Checks if `hw2/movielens-1m.zip` exists in S3  
- Downloads the dataset from GroupLens if missing  
- Uploads it to the S3 bucket  

**Output:**  
- `hw2/movielens-1m.zip`

---

### Task 2 – Create Movie Embeddings (Subset)

**Function:**  
`T2_create_embeddings(bucket_name, embeddings_name, subset_fraction)`

**Description:**  
- Loads MovieLens data from S3  
- Samples a subset of users (default 30%)  
- Generates BERT embeddings for movies rated by sampled users  
- Saves embeddings to S3  

**Output:**  
- `hw2/movie_subset_embeddings_with_ids.pt`

---

### Task 3 – Generate Recommendations (Subset)

**Function:**  
`T3_recommendations(bucket_name, embeddings_key)`

**Description:**  
- Loads subset embeddings  
- Generates recommendations for:
  - Cold user (popularity-based)
  - Top user (cosine similarity)  
- Saves results to S3  

**Output:**  
- `hw2/T3_recommendations.csv`

---

### Task 4 – Full Dataset Pipeline

**Function:**  
`T4_full(bucket_name, embeddings_name)`

**Description:**  
- Runs Task 2 with full dataset (`subset_fraction=1`)  
- Runs Task 3 using full embeddings  
- Saves full-dataset recommendations  

**Output:**  
- `hw2/FULL_movie_subset_embeddings_with_ids.pt`  
- `hw2/T4_full_recommendations.csv`

---

### Task 5 – Personalized Recommendations

**Functions:**  
`create_myprofile(prof_name, bucket_name)`  
`T5_personalized(bucket_name, embeddings_key, profile_key)`

**Description:**  
- Creates a CSV containing 10 selected movies and ratings  
- Uploads user profile to S3  
- Builds a personalized embedding  
- Recommends 5 movies  
- Saves results to S3  

**Output:**  
- `hw2/livs_profile.csv`  
- `hw2/T5_personalized_recommendations.csv`

---


## AI Usage Disclosure

### (1) Tool(s) Used
- OpenAI ChatGPT

### (2) Prompts
Prompts used during development:

- "why am I getting a FileNotFoundError when downloading from S3 to a hw2 folder?"
- "how do I structure argparse to run tasks by task number?"
- "why does torch.load give a weights_only error in PyTorch 2.6?"
- "can you help me write out a minimal README for this assignment?"

### (3) What I Changed 

- Adjusted S3 key handling to ensure all files use the `hw2/` prefix.
- Fixed local vs. S3 path confusion when downloading files.
- Corrected PyTorch `torch.load` usage to handle `weights_only=False`.
- Ensured consistent embedding key names across Tasks 2–5.

### (4) How I Tested
- Confirming successful uploads to S3
- Inspecting generated CSV outputs
- Ensuring recommendations excluded already-rated movies
- Running full pipeline (Tasks 1–5) sequentially without runtime errors


