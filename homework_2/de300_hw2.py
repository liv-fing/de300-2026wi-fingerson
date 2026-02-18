# HW2 - BERT with AWS

# imports
import boto3
from botocore.exceptions import ClientError
import requests
import zipfile
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModel
import io
import torch.nn.functional as F
from argparse import ArgumentParser
import os

### ---------------------------------------------------- Task 1 - Working

def T1_download_movielens(
        bucket_name = 'fingerson-winter26'): 
    '''
    checks if file in in given bucket
    downloads movielens1m if not already in bucket 
    uploads movielens1m to bucket
    '''
    s3 = boto3.client('s3')

    # check if file exists 
    try:
        s3.head_object(Bucket=bucket_name, Key='hw2/movielens-1m.zip')
        print('File exists in bucket. Skipping download')
        return
    except ClientError as e: # check if file not found
        if e.response['Error']['Code'] == '404':
            print('File does not exist in bucket. Will download.')
        else:
            raise
    
    # download from datarec-lib
    movielens_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    response = requests.get(movielens_url,stream=True) 
    response.raise_for_status()

    # upload to s3
    s3.upload_fileobj(response.raw, bucket_name, 'hw2/movielens-1m.zip')
    print('Uploaded')



### ---------------------------------------------------- Task 2 - Working

def T2_create_embeddings(
        bucket_name = 'fingerson-winter26', # 
        embeddings_name = 'hw2/movie_subset_embeddings_with_ids.pt', 
        subset_fraction = 0.3): # 0.3 for task 2, 1 for task 4):
    
    '''
    download movielens1m.zip from s3 bucket
    sample 30% of users
    embed movies rated by those users using BERT
    upload embeddings to s3 bucket
    '''
    # connect to s3
    s3 = boto3.client('s3')

    # check if embeddings already exist in bucket
    try:
        s3.head_object(Bucket = bucket_name, Key = embeddings_name)
        print("embeddings exist already. skip this step")
        return
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('embeddings do not exist in bucket. Will create and upload.')
        else:
            raise

    # create local hw2
    os.makedirs("hw2", exist_ok=True)


    # download zip file
    s3.download_file(bucket_name, 'hw2/movielens-1m.zip', 'hw2/movielens-1m.zip')
    print("Downloaded zip file from s3 bucket")

    # open zip file and read movies data
    zip_path = 'hw2/movielens-1m.zip'
    with zipfile.ZipFile(zip_path, "r") as z:

        print("collecting users from ratings.dat")
        users = set()
        with z.open("ml-1m/ratings.dat") as f:
            for line in f:
                uid = int(line.split(b"::", 1)[0])
                users.add(uid)

        users = sorted(users)
        rng = random.Random(2)
        sampled_users = set(rng.sample(users, max(1, int(subset_fraction * len(users)))))
        print(f"users total={len(users)}, sampled={len(sampled_users)}")

        print("collecting movie_ids for sampled users")
        movie_ids = set()
        with z.open("ml-1m/ratings.dat") as f:
            for line in f:
                uid_b, mid_b, *_ = line.split(b"::")
                if int(uid_b) in sampled_users:
                    movie_ids.add(int(mid_b))

        print("reading movies.dat")
        with z.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                f, sep="::", engine="python", header=None,
                names=["movie_id", "title", "genres"],
                encoding="latin-1",
                dtype={"movie_id": "int32"}
            )
        print("finished movies.dat")

    # subset movies to those rated by sampled users and create text field for embedding
    movie_subset = movies[movies["movie_id"].isin(movie_ids)].copy()
    movie_subset["text"] = (
        movie_subset["title"].fillna("").astype(str)
        + " [GENRES] "
        + movie_subset["genres"].fillna("").astype(str)
    )
    print(f"Number of movies embedded: {len(movie_subset)}")

    # from lab4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    bert.eval()

    @torch.no_grad() # this speeds things up 
    def encode_texts(texts, batch_size=4, max_len=32): 
        '''
        encodes list of texts to embeddings using BERT
        code from lab4
        '''
        embs = []
        n = len(texts)
        for i in range(0, n, batch_size):
            if i % (batch_size * 50) == 0:
                print(f'embedding batch starting at {i}/{n}') # check to make sure it is running not frozen
            batch = texts[i:i+batch_size]
            inp = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = bert(**inp).last_hidden_state[:,0,:]  # [CLS]
            embs.append(out.cpu())
        return torch.cat(embs, dim=0)
    
    # encode movie subset
    movie_subset_embeddings = encode_texts(movie_subset["text"].tolist())
    print("movie subset embeddings shape:", movie_subset_embeddings.shape)

    # save embeddings and movie ids together 
    payload = {
    "movie_subset_embeddings": movie_subset_embeddings,
    "movie_id": movie_subset["movie_id"].to_numpy(),  
    }
    torch.save(payload, embeddings_name)
    s3.upload_file(embeddings_name, bucket_name, embeddings_name)


### ---------------------------------------------------- Task 3 - Tested, needs edits

def T3_recommendations(
    bucket_name= 'fingerson-winter26',
    embeddings_key= 'hw2/movie_subset_embeddings_with_ids.pt', 
    out_key="hw2/T3_recommendations.csv",
    seed=2,
    top_percent=0.05,
):
    """
    Task 3:
      - Use the subset of users (same sampling scheme as Task 2; controlled by subset_fraction)
      - Recommend 5 movies for:
          1) Cold user (no history): popularity baseline
          2) Top user (random user from top 5% interactions): cosine similarity to user profile
      - Save to CSV and upload to S3.
    """
    import io
    import csv, zipfile, random
    from datetime import datetime, timezone

    import boto3
    import pandas as pd
    import torch
    import torch.nn.functional as F

    s3 = boto3.client("s3")
    rng = random.Random(seed)

    # load embeddings from S3
    obj = s3.get_object(Bucket=bucket_name, Key=embeddings_key)
    payload = torch.load(io.BytesIO(obj['Body'].read()), map_location="cpu", weights_only=False)

    item_emb = payload["movie_subset_embeddings"]  # shape [N, d]
    emb_movie_ids = [int(x) for x in list(payload["movie_id"])]
    emb_movie_id_set = set(emb_movie_ids)


    # normalize for cosine similarity
    E = F.normalize(item_emb, p=2, dim=1)
    movieid_to_row = {mid: i for i, mid in enumerate(emb_movie_ids)}

    # load zip 
    zip_bytes = s3.get_object(Bucket=bucket_name, Key="hw2/movielens-1m.zip")["Body"].read()
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")

    # Load movies.dat (titles/genres) 
    with zf.open("ml-1m/movies.dat") as f:
        movies = pd.read_csv(
            f, sep="::", engine="python", header=None,
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
            dtype={"movie_id": "int32"}
        )
    mid_to_title = dict(zip(movies["movie_id"].astype(int), movies["title"].astype(str)))
    mid_to_genres = dict(zip(movies["movie_id"].astype(int), movies["genres"].astype(str)))
    
    # Load ratings for all users, but only movies with embeddings
    rows = []
    with zf.open("ml-1m/ratings.dat") as f:
        for line in f:
            uid_b, mid_b, rating_b, ts_b = line.split(b"::") # user id, movie id, rating, timestamp
            uid = int(uid_b)
            mid = int(mid_b)
            if mid not in emb_movie_id_set: # filter to only use movies with embeddings
                continue
            rating = int(rating_b)
            ts = int(ts_b)
            rows.append((uid, mid, rating, ts))

    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    if ratings.empty:
        raise RuntimeError("No ratings left after applying subset + embedding-scope filters.")

    # cold user, use top 5 movies by popularity
    pop = ratings["movie_id"].value_counts() # popularity by count of ratings
    cold_top5 = pop.head(5).index.tolist()

    # top user, random user among top 5% by interaction count
    user_counts = ratings["user_id"].value_counts()  # sorted desc
    k = max(1, int(len(user_counts) * top_percent))
    top_users = user_counts.head(k).index.tolist()
    top_user_id = rng.choice(top_users)

    N = len(user_counts)
    rank_series = user_counts.rank(method="first", ascending=False)
    top_user_rank = int(rank_series.loc[top_user_id])
    top_user_percentile = 1 - (top_user_rank - 1) / N

    top_user_rows = ratings[ratings["user_id"] == top_user_id].copy()
    last_ts = int(top_user_rows["timestamp"].max())

    mids = top_user_rows["movie_id"].astype(int).tolist()
    rows_idx = [movieid_to_row[m] for m in mids if m in movieid_to_row]
    if len(rows_idx) == 0:
        raise RuntimeError("Top user has no movies in embedding universe (unexpected).")

    u = E[rows_idx].mean(dim=0, keepdim=True)  # [1, d]
    sims = (E @ u[0])  # [num_movies]

    # filter already seen (by embedding row index), then take top-5
    seen_rows = set(rows_idx)
    candidates = [i for i in torch.argsort(sims, descending=True).tolist() if i not in seen_rows]
    top = candidates[:5]
    recs = [emb_movie_ids[i] for i in top]

    if len(recs) < 5:
        rated_set = set(mids)
        for mid in pop.index.astype(int).tolist():
            if mid not in rated_set and mid not in recs:
                recs.append(mid)
            if len(recs) == 5:
                break

    # Assemble clean output rows
    def format_ids(movie_ids_5):
        return " | ".join(str(mid) for mid in movie_ids_5)

    def format_titles(movie_ids_5):
        return " | ".join(mid_to_title.get(mid, "") for mid in movie_ids_5)


    out_rows = [
        {
            "User_ID": "",
            "User_Type": "cold_user",
            "Last_Interaction_Time": "",
            "Num_User_Interactions": 0,
            "Recommended_Movies": format_ids(cold_top5),
            "Recommended_Movie_Titles": format_titles(cold_top5),
        },
        {
            "User_ID": int(top_user_id),
            "User_Type": "top_user",
            "Top_User_Percentile": round(float(top_user_percentile),4),
            "Last_Interaction_Time": int(last_ts),
            "Num_User_Interactions": int(user_counts.loc[top_user_id]),
            "Recommended_Movies": format_ids(recs),
            "Recommended_Movie_Titles": format_titles(recs),
        },
    ]


    # Write CSV locally and upload to S3
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_key, index=False)


    s3.upload_file(out_key, bucket_name, out_key)
    print(f"Uploaded recommendations to s3://{bucket_name}{out_key}")
    print(df_out)



## Task 4

# just run T2 with different embeddings name and subset_fraction=1 
# then, run T3 with those embeddings to get new predictions

def T4_full(bucket_name='fingerson-winter26', 
            embeddings_name='hw2/FULL_movie_subset_embeddings_with_ids.pt'):

    # embeddings for full dataset
    T2_create_embeddings(
        bucket_name = bucket_name,
        embeddings_name=embeddings_name, 
        subset_fraction=1)
    
    # recommendations using full dataset embeddings
    T3_recommendations(
        bucket_name=bucket_name,
        embeddings_key=embeddings_name, 
        out_key='hw2/T4_full_recommendations.csv',
        seed=3)

def create_myprofile(prof_name = 'hw2/livs_profile.csv', bucket_name = 'fingerson-winter26'):
    '''
    my own movies and ratings
    add to s3 bucket if not already there
    '''

    s3 = boto3.client('s3')

    # check if file exists
    try:
        s3.head_object(Bucket=bucket_name, Key=prof_name)
        print('Profile already exists in bucket. Skipping creation.')
        return
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('profile does not exist in bucket. creating and uploading.')
        else:
            raise

    # add ratings
    my_ratings = [
    (296, 5),
    (2571, 4),
    (2021, 3),
    (3034, 4),
    (2495, 4),
    (2810, 3),
    (3897, 5),
    (1582, 4),
    (592, 3),
    (3702, 4),]

    df = pd.DataFrame(my_ratings, columns=["movie_id", "rating"])
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(int)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket_name, Key=prof_name, Body=csv_bytes)

    print(f"Uploaded s3://{bucket_name}/{prof_name}")
    print(df)

def T5_personalized(bucket_name='fingerson-winter26', 
                    num_recs=5,
                    embeddings_key='hw2/FULL_movie_subset_embeddings_with_ids.pt',
                    profile_key='hw2/livs_profile.csv',
                    out_key='hw2/T5_personalized_recommendations.csv'):
    
    '''
    load profile csv from s3
    use precomputed embeddings
    recommend movies and upload results to s3
    '''

    s3 = boto3.client('s3')

    # load profile from s3
    obj = s3.get_object(Bucket=bucket_name, Key=profile_key)
    prof_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    prof_df["movie_id"] = prof_df["movie_id"].astype(int)
    prof_df["rating"] = prof_df["rating"].astype(int)

    if len(prof_df) != 10:
        print(f"expecting 10 rating but there are {len(prof_df)}") # check

    # load embeddings
    obj = s3.get_object(Bucket=bucket_name, Key=embeddings_key)
    payload = torch.load(io.BytesIO(obj['Body'].read()), map_location="cpu", weights_only=False)
    embed_raw = payload["movie_subset_embeddings"]  # shape [N, d]
    emb_movie_ids = [int(x) for x in list(payload["movie_id"])]
    movie_id_to_row = {mid: i for i, mid in enumerate(emb_movie_ids)}
    emb_movie_id_set = set(emb_movie_ids)
    E = F.normalize(embed_raw, p=2, dim=1)

    # build user vector
    chosen_ids = prof_df['movie_id'].tolist()
    rows_idx = [movie_id_to_row[mid] for mid in chosen_ids]
    u = E[rows_idx].mean(dim=0, keepdim=True)   # [1, d]
    u = F.normalize(u, p=2, dim=1)

    sims = (E @ u[0]) # cos similarity
    
    seen_rows = set(rows_idx)
    candidates = [i for i in torch.argsort(sims, descending=True).tolist() if i not in seen_rows]
    top_rows = candidates[:num_recs]
    rec_ids = [emb_movie_ids[i] for i in top_rows]

    # Load movies metadata (same as Task 3)
    zip_bytes = s3.get_object(Bucket=bucket_name, Key="hw2/movielens-1m.zip")["Body"].read()
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")

    with zf.open("ml-1m/movies.dat") as f:
        movies = pd.read_csv(
            f,
            sep="::",
            engine="python",
            header=None,
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
            dtype={"movie_id": "int32"},
        )
    mid_to_title = dict(zip(movies["movie_id"].astype(int), movies["title"].astype(str)))

    out_df = pd.DataFrame(
        [{ 
            "User_ID": "my_profile",
            "Profile_Key": profile_key,
            "Chosen_Movies": " | ".join(str(mid) for mid in chosen_ids),
            "Chosen_Ratings": " | ".join(str(r) for r in prof_df['rating'].tolist()),
            "Recommended_Movies": " | ".join(str(mid) for mid in rec_ids),
            "Recommended_Movie_Titles": " | ".join(mid_to_title.get(int(mid), "") for mid in rec_ids),
        }])

    csv_byes = out_df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket_name, Key=out_key, Body=csv_byes)

    print(f"uploaded personalized recs to s3://{bucket_name}/hw2/{out_key}")
    print(out_df)
    return out_df






### ARG PARSE

def main(
        tasknum,
        bucketname,
        task2_subset_fraction,
        task2_embeddings_name, 
        task4_embeddings_name,
        task5_profile_name,
        ):
    '''
    main function to run any of the 5 tasks based on tasknum
    '''
    
    if tasknum == 1:
        T1_download_movielens(bucket_name=bucketname)
    elif tasknum == 2:
        T2_create_embeddings(bucket_name=bucketname, embeddings_name=task2_embeddings_name, subset_fraction=task2_subset_fraction)
    elif tasknum == 3:
        T3_recommendations(bucket_name=bucketname, embeddings_key=task2_embeddings_name)
    elif tasknum == 4:
        T4_full(bucket_name=bucketname, embeddings_name=task4_embeddings_name)
    elif tasknum == 5:
        create_myprofile(prof_name=task5_profile_name, bucket_name=bucketname)
        T5_personalized(bucket_name=bucketname, embeddings_key=task4_embeddings_name, profile_key=task5_profile_name)
    else:
        raise ValueError("Invalid task number. Must be 1-5.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recommender with BERT")

    parser.add_argument('--tasknum', type=int, default=1, help='Task number (1-5)')
    parser.add_argument('--bucketname', type=str, default='fingerson-winter26', help='S3 bucket name (default: fingerson-winter26)')
    parser.add_argument('--task2_subset_fraction', type=float, default=0.3, help='Subset fraction for Task 2 (default: 0.3)')
    parser.add_argument('--task2_embeddings_name', type=str, default='hw2/movie_subset_embeddings_with_ids.pt', help='S3 key for Task 2 embeddings (default: movie_subset_embeddings_with_ids.pt)')
    parser.add_argument('--task4_embeddings_name', type=str, default='hw2/FULL_movie_subset_embeddings_with_ids.pt', help='S3 key for Task 4 embeddings (default: FULL_movie_subset_embeddings_with_ids.pt)')
    parser.add_argument('--task5_profile_name', type=str, default='hw2/livs_profile.csv', help='name for user profile')
    args = parser.parse_args()

    main(
        tasknum = args.tasknum,
        bucketname = args.bucketname,
        task2_subset_fraction = args.task2_subset_fraction,
        task2_embeddings_name = args.task2_embeddings_name,
        task4_embeddings_name= args.task4_embeddings_name,
        task5_profile_name=args.task5_profile_name)