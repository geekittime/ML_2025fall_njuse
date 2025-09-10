import pandas as pd

if __name__ == "__main__":
    data_folder = 'yii2'
    pr_info = pd.read_excel(f'{data_folder}/PR_info_add_conversation.xlsx')
    pr_features = pd.read_excel(f'{data_folder}/PR_features.xlsx')
    author_features = pd.read_excel(f'{data_folder}/author_features.xlsx')
    pr_commits = pd.read_excel(f'{data_folder}/PR_commit_info.xlsx')
    pr_comments = pd.read_excel(f'{data_folder}/PR_comment_info.xlsx')

    # Filter closed pull requests
    pr_info = pr_info[pr_info['state'] == 'closed']

    # Delete useless columns
    del pr_info['author']
    del pr_info['state']
    del pr_info['title']
    del pr_info['body']

    # Canonicalize datetime columns
    datetime_cols = ['created_at', 'updated_at', 'merged_at', 'closed_at']
    for col in datetime_cols:
        pr_info[col] = pd.to_datetime(pr_info[col], utc=True)
        pr_info[col] = pr_info[col].dt.tz_localize(None)

    # Extract commit max changes
    pr_with_commits = pd.merge(pr_info, pr_commits, left_on='number', right_on='belong_to_PR', how='left').groupby(
        'number')
    pr_commit_max = pr_with_commits[['changes', 'additions_y', 'deletions_y', 'segs', 'add_segs', 'del_segs']].max()
    pr_commit_max = pr_commit_max.rename(
        columns={'changes': 'max_changes', 'additions_y': 'max_additions', 'deletions_y': 'max_deletions',
                 'segs': 'max_segs', 'add_segs': 'max_add_segs', 'del_segs': 'max_del_segs'})
    pr_info = pr_info.merge(pr_commit_max, on='number')

    # Extract useful features from pr_features
    pr_features = pr_features[
        ['number', 'has_test', 'has_feature', 'has_bug', 'has_document', 'has_improve', 'has_refactor', 'title_length',
         'title_readability', 'body_length', 'body_readability', 'segs_added', 'segs_deleted', 'segs_updated',
         'files_added', 'files_deleted', 'files_updated', 'modify_proportion', 'modify_entropy', 'test_churn',
         'non_test_churn', 'reviewer_num', 'bot_reviewer_num', 'is_reviewed', 'comment_length']]
    pr_info = pr_info.merge(pr_features, on='number')

    # Extract useful features from author_features
    author_features = author_features[
        ['number', 'experience', 'is_reviewer', 'participation', 'changes_per_week', 'avg_round', 'avg_duration',
         'merge_proportion']]
    pr_info = pr_info.merge(author_features, on='number')

    # Sort by PR number
    pr_info = pr_info.sort_values('number')

    pr_info.to_excel(f'{data_folder}/PR_extracted_features.xlsx', index=False)