import pandas as pd

if __name__ == "__main__":
    data_folder = 'yii2'
    pr_info = pd.read_excel(f'{data_folder}/PR_info_add_conversation.xlsx')
    pr_features = pd.read_excel(f'{data_folder}/PR_features.xlsx')
    author_features = pd.read_excel(f'{data_folder}/author_features.xlsx')
    pr_comments = pd.read_excel(f'{data_folder}/PR_comment_info.xlsx')

    # Filter closed pull requests
    pr_info = pr_info[pr_info['state'] == 'closed']

    # Delete useless columns
    pr_info = pr_info[["number", "created_at", "updated_at", "merged_at", "closed_at", "merged", "additions", "deletions"]]

    # Canonicalize datetime columns
    datetime_cols = ['created_at', 'updated_at', 'merged_at', 'closed_at']
    for col in datetime_cols:
        pr_info[col] = pd.to_datetime(pr_info[col], utc=True)
        pr_info[col] = pr_info[col].dt.tz_localize(None)

    # Extract last pr update time
    pr_info['last_pr_update'] = (pr_info['updated_at'] - pr_info['created_at']).dt.total_seconds() / 3600;

    # Extract useful features from pr_features
    pr_features = pr_features[
        ['number', 'title_length', 'body_length', 'files_added', 'files_deleted', 'files_updated']]
    pr_info = pr_info.merge(pr_features, on='number')

    # Extract useful features from author_features
    author_features = author_features[['number', 'changes_per_week', 'merge_proportion']]
    pr_info = pr_info.merge(author_features, on='number')

    # Extract last comment update time
    pr_comments['updated_at'] = pd.to_datetime(pr_comments['updated_at'], utc=True).dt.tz_localize(None)
    pr_with_comments = pd.merge(pr_info, pr_comments, left_on='number', right_on='belong_to_PR', how='left')
    pr_with_comments['last_comment_update'] = (pr_with_comments['updated_at_y'] - pr_with_comments['created_at_x']).dt.total_seconds() / 3600;
    pr_comment_last = pr_with_comments.groupby('number')['last_comment_update'].max()
    pr_info = pr_info.merge(pr_comment_last, on='number')

    # Sort by PR number
    pr_info = pr_info.sort_values('number')

    pr_info.to_excel(f'{data_folder}/PR_extracted_features.xlsx', index=False)