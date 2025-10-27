import pandas as pd

if __name__ == "__main__":
    data_folder = input("Folder to extract features from: ")
    pr_list = pd.read_csv(f'{data_folder}/pr_list.csv')
    pr_details = pd.read_csv(f'{data_folder}/pr_details.csv')
    pr_reviews = pd.read_csv(f'{data_folder}/pr_reviews.csv')
    contributors = pd.read_csv(f'{data_folder}/contributors.csv')

    # Filter closed pull requests
    pr_list = pr_list[pr_list['state'] == 'closed']

    # Canonicalize datetime columns
    datetime_cols = ['created_at', 'updated_at', 'merged_at', 'closed_at']
    for col in datetime_cols:
        pr_list[col] = pd.to_datetime(pr_list[col], utc=True)
        pr_list[col] = pr_list[col].dt.tz_localize(None)

    # Calculate merged
    pr_list["merged"] = ~pr_list["merged_at"].isna()

    # Extract useful columns
    pr_list["title_length"] = pr_list["title"].str.len()
    pr_list["body_length"] = pr_list["body"].str.len()
    pr_list["body_length"] = pr_list["body_length"].fillna(0)
    pr_list = pr_list[["number", "user/login", "title_length", "body_length", "created_at", "updated_at", "merged_at", "closed_at", "merged"]]

    merge_proportion = pr_list[["user/login", "merged"]].groupby("user/login")["merged"].mean()
    merge_proportion = merge_proportion.rename("merge_proportion")
    pr_list = pr_list.merge(merge_proportion, on="user/login")

    # Extract last pr update time
    pr_list['last_pr_update'] = (pr_list['updated_at'] - pr_list['created_at']).dt.total_seconds() / 3600

    # Extract additions, deletions and changed files from pr details
    pr_list = pr_list.merge(pr_details[["number", "additions", "deletions", "changed_files"]], on="number")

    # Extract useful features from contributors
    contributors = contributors[["login", "contributions"]]
    pr_list = pd.merge(pr_list, contributors, left_on="user/login", right_on="login", how="left")
    del pr_list["login"]
    pr_list["contributions"] = pr_list["contributions"].fillna(0)

    # Extract last comment update time
    pr_reviews['submitted_at'] = pd.to_datetime(pr_reviews['submitted_at'], utc=True).dt.tz_localize(None)
    pr_with_comments = pd.merge(pr_list, pr_reviews, left_on='number', right_on='pr_number', how='left')
    pr_with_comments['last_comment_update'] = (pr_with_comments['submitted_at'] - pr_with_comments['created_at']).dt.total_seconds() / 3600;
    pr_comment_last = pr_with_comments.groupby('number')['last_comment_update'].max()
    pr_list = pr_list.merge(pr_comment_last, on='number')

    # Remove user column
    del pr_list["user/login"]

    # Sort by PR number
    pr_list = pr_list.sort_values('number')

    pr_list.to_excel(f'{data_folder}/PR_extracted_features.xlsx', index=False)