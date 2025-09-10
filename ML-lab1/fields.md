# GITHUB 各 Endpoint 需要获取的字段

## 获取 PR 列表

`/repos/{owner}/{repo}/pulls?state={state}`其中`state`可取`all`, `open`, `closed`

示例JSON

```json
[
  {
    "url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905",
    "id": 1442030866,
    "node_id": "PR_kwDOADRbGc5V86US",
    "html_url": "https://github.com/yiisoft/yii2/pull/19905",
    "diff_url": "https://github.com/yiisoft/yii2/pull/19905.diff",
    "patch_url": "https://github.com/yiisoft/yii2/pull/19905.patch",
    "issue_url": "https://api.github.com/repos/yiisoft/yii2/issues/19905",
    "number": 19905,
    "state": "open",
    "locked": false,
    "title": "Fix Gridview or Listview 'maxButtonCount' attribute is not working wh…",
    "user": {
      "login": "msobin",
      ...
    },
    "body": "…en it's assign with 2 #19655\r\n\r\n| Q             | A\r\n| ------------- | ---\r\n| Is bugfix?    | ✔️\r\n| New feature?  | ❌\r\n| Breaks BC?    | ❌\r\n| Fixed issues  | <!-- comma-separated list of tickets # fixed by the PR, if any -->\r\n",
    "created_at": "2023-07-20T02:57:04Z",
    "updated_at": "2025-05-29T11:01:43Z",
    "closed_at": null,
    "merged_at": null,
    "merge_commit_sha": null,
    "assignee": null,
    "assignees": [],
    "requested_reviewers": [],
    "requested_teams": [...],
    "labels": [...],
    "milestone": null,
    "draft": false,
    "commits_url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905/commits",
    "review_comments_url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905/comments",
    "review_comment_url": "https://api.github.com/repos/yiisoft/yii2/pulls/comments{/number}",
    "comments_url": "https://api.github.com/repos/yiisoft/yii2/issues/19905/comments",
    "statuses_url": "https://api.github.com/repos/yiisoft/yii2/statuses/598a045ec8c0da26ab699a1322f84d281e88c1e6",
    "head": {...},
    "base": {
      "label": "yiisoft:master",
      "ref": "master",
      "sha": "53256fdd241fd842d7f1c650de933beb7973fc35",
      "user": {...},
      "repo": {
        "id": 3431193,
        "node_id": "MDEwOlJlcG9zaXRvcnkzNDMxMTkz",
        "name": "yii2",
        "full_name": "yiisoft/yii2",
        "private": false,
        "owner": {...},
        "html_url": "https://github.com/yiisoft/yii2",
        "description": "Yii 2: The Fast, Secure and Professional PHP Framework",
        "fork": false,
        ...
        "created_at": "2012-02-13T15:32:36Z",
        "updated_at": "2025-09-10T09:10:13Z",
        "pushed_at": "2025-09-08T10:08:53Z",
        ...
        "size": 84248,
        "stargazers_count": 14288,
        "watchers_count": 14288,
        "language": "PHP",
        "has_issues": true,
        "has_projects": false,
        "has_downloads": true,
        "has_wiki": true,
        "has_pages": false,
        "has_discussions": true,
        "forks_count": 6871,
        "mirror_url": null,
        "archived": false,
        "disabled": false,
        "open_issues_count": 523,
        "license": {...},
        "allow_forking": true,
        "is_template": false,
        "web_commit_signoff_required": false,
        "topics": [
          "framework",
          "hacktoberfest",
          "php",
          "php-framework",
          "yii",
          "yii2"
        ],
        "visibility": "public",
        "forks": 6871,
        "open_issues": 523,
        "watchers": 14288,
        "default_branch": "master"
      }
    },
    "_links": {...},
    "author_association": "NONE",
    "auto_merge": null,
    "active_lock_reason": null
  }
]
```

需要获取的字段
- number
- state
- title
- user/login
- body
- created_at
- updated_at
- closed_at
- merged_at
- base/repo/description
- base/repo/created_at
- base/repo/updated_at
- base/repo/pushed_at
- base/repo/size
- base/repo/stargazers_count
- base/repo/language
- base/repo/open_issues_count

## 获取 PR 详细信息

`/repos/{owner}/{repo}/pulls/{pr_number}`

示例JSON

```json
{
  "url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905",
  "id": 1442030866,
  "node_id": "PR_kwDOADRbGc5V86US",
  "html_url": "https://github.com/yiisoft/yii2/pull/19905",
  "diff_url": "https://github.com/yiisoft/yii2/pull/19905.diff",
  "patch_url": "https://github.com/yiisoft/yii2/pull/19905.patch",
  "issue_url": "https://api.github.com/repos/yiisoft/yii2/issues/19905",
  "number": 19905,
  "state": "open",
  "locked": false,
  "title": "Fix Gridview or Listview 'maxButtonCount' attribute is not working wh…",
  "user": {
    "login": "msobin",
    ...
  },
  "body": "…en it's assign with 2 #19655\r\n\r\n| Q             | A\r\n| ------------- | ---\r\n| Is bugfix?    | ✔️\r\n| New feature?  | ❌\r\n| Breaks BC?    | ❌\r\n| Fixed issues  | \u003C!-- comma-separated list of tickets # fixed by the PR, if any --\u003E\r\n",
  "created_at": "2023-07-20T02:57:04Z",
  "updated_at": "2025-05-29T11:01:43Z",
  "closed_at": null,
  "merged_at": null,
  "merge_commit_sha": null,
  "assignee": null,
  "assignees": [],
  "requested_reviewers": [],
  "requested_teams": [...],
  "labels": [...],
  "milestone": null,
  "draft": false,
  "commits_url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905/commits",
  "review_comments_url": "https://api.github.com/repos/yiisoft/yii2/pulls/19905/comments",
  "review_comment_url": "https://api.github.com/repos/yiisoft/yii2/pulls/comments{/number}",
  "comments_url": "https://api.github.com/repos/yiisoft/yii2/issues/19905/comments",
  "statuses_url": "https://api.github.com/repos/yiisoft/yii2/statuses/598a045ec8c0da26ab699a1322f84d281e88c1e6",
  "head": {...},
  "base": {...},
  "_links": {...},
  "author_association": "NONE",
  "auto_merge": null,
  "active_lock_reason": null,
  "merged": false,
  "mergeable": null,
  "rebaseable": null,
  "mergeable_state": "unknown",
  "merged_by": null,
  "comments": 3,
  "review_comments": 0,
  "maintainer_can_modify": true,
  "commits": 6,
  "additions": 5,
  "deletions": 3,
  "changed_files": 2
}
```

需要获取的字段
- number
- state
- title
- user/login
- body
- created_at
- updated_at
- closed_at
- merged_at
- merged
- comments
- review_comments
- commits
- additions
- deletions
- changed_files

## 获取修改文件列表

`/repos/{owner}/{repo}/pulls/{pr_number}/files`

示例JSON

```json
[
  {
    "sha": "ea5f295d0bcccf63e28239417a50635d60025c44",
    "filename": "framework/CHANGELOG.md",
    "status": "modified",
    "additions": 1,
    "deletions": 0,
    "changes": 1,
    "blob_url": "https://github.com/yiisoft/yii2/blob/598a045ec8c0da26ab699a1322f84d281e88c1e6/framework%2FCHANGELOG.md",
    "raw_url": "https://github.com/yiisoft/yii2/raw/598a045ec8c0da26ab699a1322f84d281e88c1e6/framework%2FCHANGELOG.md",
    "contents_url": "https://api.github.com/repos/yiisoft/yii2/contents/framework%2FCHANGELOG.md?ref=598a045ec8c0da26ab699a1322f84d281e88c1e6",
    "patch": "@@ -13,6 +13,7 @@ Yii Framework 2 Change Log\n - Enh #19853: Added support for default value for `\\yii\\helpers\\Console::select()` (rhertogh)\n - Bug #19868: Added whitespace sanitation for tests, due to updates in ICU 72 (schmunk42)\n - Enh #19884: Added support Enums in Query Builder (sk1t0n)\n+- Bug #19655: Fix Gridview or Listview 'maxButtonCount' attribute is not working when it's assign with 2 (koktut)\n - Bug #19906: Fixed multiline strings in the `\\yii\\console\\widgets\\Table` widget (rhertogh)\n \n "
  }
]
```

需要获取的字段
- sha
- filename
- additions
- deletions
- changes
- patch

## 获取评论与审核信息

`/repos/{owner}/{repo}/pulls/{pr_number}/reviews`

示例JSON

```json
[
  {
    "id": 3134716817,
    "node_id": "PRR_kwDOADRbGc661_uR",
    "user": {
      "login": "max-s-lab",
      "id": 63721828,
      ...
    },
    "body": "",
    "state": "COMMENTED",
    "html_url": "https://github.com/yiisoft/yii2/pull/20491#pullrequestreview-3134716817",
    "pull_request_url": "https://api.github.com/repos/yiisoft/yii2/pulls/20491",
    "author_association": "CONTRIBUTOR",
    "_links": {...},
    "submitted_at": "2025-08-20T04:06:52Z",
    "commit_id": "882b5e85ed100eb67586c4f38665a1e808d20735"
  },
  {
    "id": 3137829659,
    "node_id": "PRR_kwDOADRbGc67B3sb",
    "user": {
      "login": "samdark",
      "id": 47294,
      ...
    },
    "body": "Looks :+1: Please add a line for CHANGELOG.",
    "state": "APPROVED",
    "html_url": "https://github.com/yiisoft/yii2/pull/20491#pullrequestreview-3137829659",
    "pull_request_url": "https://api.github.com/repos/yiisoft/yii2/pulls/20491",
    "author_association": "MEMBER",
    "_links": {...},
    "submitted_at": "2025-08-20T18:47:26Z",
    "commit_id": "0a64c2ada555ce53457bd7c9d8b48cc47febfc13"
  }
]
```

需要获取的字段
- id
- user/login
- body
- submitted_at

## 提取作者信息

`/repos/{owner}/{repo}/contributors`

示例JSON

```json
[
  {
    "login": "samdark",
    "id": 47294,
    "node_id": "MDQ6VXNlcjQ3Mjk0",
    "avatar_url": "https://avatars.githubusercontent.com/u/47294?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/samdark",
    "html_url": "https://github.com/samdark",
    "followers_url": "https://api.github.com/users/samdark/followers",
    "following_url": "https://api.github.com/users/samdark/following{/other_user}",
    "gists_url": "https://api.github.com/users/samdark/gists{/gist_id}",
    "starred_url": "https://api.github.com/users/samdark/starred{/owner}{/repo}",
    "subscriptions_url": "https://api.github.com/users/samdark/subscriptions",
    "organizations_url": "https://api.github.com/users/samdark/orgs",
    "repos_url": "https://api.github.com/users/samdark/repos",
    "events_url": "https://api.github.com/users/samdark/events{/privacy}",
    "received_events_url": "https://api.github.com/users/samdark/received_events",
    "type": "User",
    "user_view_type": "public",
    "site_admin": false,
    "contributions": 3594
  }
]
```

需要获取的字段
- login
- contributions