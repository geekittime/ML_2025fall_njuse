"""
GitHub 仓库数据爬取脚本
1. PR 列表 (/pulls)              -> pr_list.csv
2. PR 详情 (/pulls/{number})      -> pr_details.csv
3. 修改文件 (/pulls/{number}/files) -> pr_files.csv
4. 审核评论 (/pulls/{number}/reviews) -> pr_reviews.csv
5. 贡献者 (/contributors)         -> contributors.csv

使用说明:
python repo_crawler.py --owner yiisoft --repo yii2 --state all --max-prs 200 --out-dir output
可通过环境变量 GITHUB_TOKEN 或参数 --token 传入 Personal Access Token (推荐, 防止频繁 rate limit)。
"""
import os
import sys
import csv
import time
import argparse
import logging
from typing import List, Dict, Any, Optional, Iterable
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_ROOT = "https://api.github.com"
REQUIRED_PR_LIST_FIELDS = [
    "number", "state", "title", ("user", "login"), "body",
    "created_at", "updated_at", "closed_at", "merged_at"
]
REQUIRED_PR_DETAIL_FIELDS = [
    "number", "state", "title", ("user", "login"), "body",
    "created_at", "updated_at", "closed_at", "merged_at",
    "merged", "comments", "review_comments", "commits",
    "additions", "deletions", "changed_files"
]
REQUIRED_FILE_FIELDS = ["sha", "filename", "additions", "deletions", "changes", "patch"]
REQUIRED_REVIEW_FIELDS = ["id", ("user", "login"), "body", "submitted_at"]
REQUIRED_CONTRIBUTOR_FIELDS = ["login", "contributions"]

LOGGER = logging.getLogger("repo_crawler")

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(asctime)s] %(levelname)s: %(message)s')


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", 'User-Agent': 'RepoCrawler/1.0'}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_get(session: requests.Session, url: str, params: Dict[str, Any] = None, retry: int = 3, backoff: float = 2.0) -> requests.Response:
    for attempt in range(1, retry + 1):
        resp = session.get(url, params=params)
        if resp.status_code == 403 and 'X-RateLimit-Remaining' in resp.headers:
            remaining = resp.headers.get('X-RateLimit-Remaining')
            if remaining == '0':
                reset = int(resp.headers.get('X-RateLimit-Reset', '0'))
                sleep_sec = max(0, reset - int(time.time()) + 1)
                LOGGER.warning(f"Rate limit reached. Sleep {sleep_sec}s until reset.")
                time.sleep(sleep_sec)
                continue
        if resp.status_code >= 500:
            LOGGER.warning(f"Server error {resp.status_code}, attempt {attempt}/{retry}")
            time.sleep(backoff * attempt)
            continue
        if resp.status_code == 404:
            LOGGER.error(f"Not found: {url}")
        return resp
    return resp  # last response


def paginate(session: requests.Session, url: str, params: Dict[str, Any] = None) -> Iterable[Dict[str, Any]]:
    page = 1
    while True:
        merged_params = dict(params or {})
        merged_params.update({"page": page, "per_page": 100})
        resp = github_get(session, url, merged_params)
        if resp.status_code != 200:
            LOGGER.error(f"Failed {url} status={resp.status_code} body={resp.text[:200]}")
            break
        data = resp.json()
        if not data:
            break
        for item in data:
            yield item
        if len(data) < 100:
            break
        page += 1


def extract_fields(item: Dict[str, Any], spec: List[Any]) -> Dict[str, Any]:
    out = {}
    for field in spec:
        if isinstance(field, tuple):  # nested path
            cur = item
            path = []
            for part in field:
                path.append(part)
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            out['/'.join(field)] = cur
        else:
            out[field] = item.get(field)
    return out


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_csv(filepath: str, rows: List[Dict[str, Any]]):
    if not rows:
        LOGGER.info(f"No data for {filepath}")
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    LOGGER.info(f"Wrote {filepath} ({len(rows)} rows)")


def fetch_pr_list(session: requests.Session, owner: str, repo: str, state: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    url = f"{API_ROOT}/repos/{owner}/{repo}/pulls"
    rows = []
    for item in paginate(session, url, params={"state": state, "sort": "created", "direction": "desc"}):
        rows.append(extract_fields(item, REQUIRED_PR_LIST_FIELDS))
        if limit and len(rows) >= limit:
            break
    return rows


def fetch_pr_details(session: requests.Session, owner: str, repo: str, pr_numbers: List[int], max_workers: int = 8) -> List[Dict[str, Any]]:
    """并发抓取 PR 详情"""
    rows: List[Dict[str, Any]] = []
    total = len(pr_numbers)
    lock = threading.Lock()

    def worker(num):
        url = f"{API_ROOT}/repos/{owner}/{repo}/pulls/{num}"
        resp = github_get(session, url)
        if resp.status_code != 200:
            LOGGER.error(f"Failed detail PR #{num}: {resp.status_code}")
            return None
        return extract_fields(resp.json(), REQUIRED_PR_DETAIL_FIELDS)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(worker, n): n for n in pr_numbers}
        for fut in as_completed(future_map):
            result = fut.result()
            if result:
                rows.append(result)
            with lock:
                completed += 1
                print(f"{completed}/{total} Fetched details", end='\r')
    return rows


def fetch_pr_files(session: requests.Session, owner: str, repo: str, pr_numbers: List[int], max_workers: int = 8) -> List[Dict[str, Any]]:
    """并发抓取 PR 修改文件列表"""
    rows: List[Dict[str, Any]] = []
    total = len(pr_numbers)
    lock = threading.Lock()

    def worker(num):
        url = f"{API_ROOT}/repos/{owner}/{repo}/pulls/{num}/files"
        local_rows = []
        for item in paginate(session, url):
            record = extract_fields(item, REQUIRED_FILE_FIELDS)
            record['pr_number'] = num
            local_rows.append(record)
        return local_rows

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(worker, n): n for n in pr_numbers}
        for fut in as_completed(future_map):
            result = fut.result()
            if result:
                rows.extend(result)
            with lock:
                completed += 1
                print(f"{completed}/{total} Fetched files", end='\r')
    return rows


def fetch_pr_reviews(session: requests.Session, owner: str, repo: str, pr_numbers: List[int], max_workers: int = 8) -> List[Dict[str, Any]]:
    """并发抓取 PR 审核评论"""
    rows: List[Dict[str, Any]] = []
    total = len(pr_numbers)
    lock = threading.Lock()

    def worker(num):
        url = f"{API_ROOT}/repos/{owner}/{repo}/pulls/{num}/reviews"
        local_rows = []
        for item in paginate(session, url):
            record = extract_fields(item, REQUIRED_REVIEW_FIELDS)
            record['pr_number'] = num
            local_rows.append(record)
        return local_rows

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(worker, n): n for n in pr_numbers}
        for fut in as_completed(future_map):
            result = fut.result()
            if result:
                rows.extend(result)
            with lock:
                completed += 1
                print(f"{completed}/{total} Fetched reviews", end='\r')
    return rows


def fetch_contributors(session: requests.Session, owner: str, repo: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    url = f"{API_ROOT}/repos/{owner}/{repo}/contributors"
    rows = []
    for i, item in enumerate(paginate(session, url)):
        rows.append(extract_fields(item, REQUIRED_CONTRIBUTOR_FIELDS))
        if limit and len(rows) >= limit:
            break
        print(f"{i} Fetched", end='\r')
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="GitHub 仓库 PR/文件/评论/贡献者爬取脚本")
    p.add_argument('--owner', required=True, help='仓库 owner')
    p.add_argument('--repo', required=True, help='仓库名称')
    p.add_argument('--state', default='all', choices=['all', 'open', 'closed'], help='PR 状态')
    p.add_argument('--token', help='GitHub Token (可选, 否则使用环境变量 GITHUB_TOKEN)')
    p.add_argument('--max-prs', type=int, help='限定最多抓取多少个 PR (按最新)')
    p.add_argument('--limit-contrib', type=int, help='限定贡献者数量 (可选)')
    p.add_argument('-o', '--out-dir', default='output', help='输出目录')
    p.add_argument('--skip-list', action='store_true', help='跳过 PR 列表获取')
    p.add_argument('--skip-details', action='store_true', help='跳过 PR 详情')
    p.add_argument('--skip-files', action='store_true', help='跳过 PR 修改文件')
    p.add_argument('--skip-reviews', action='store_true', help='跳过 PR 审核评论')
    p.add_argument('--skip-contrib', action='store_true', help='跳过贡献者')
    p.add_argument('--verbose', action='store_true', help='调试日志')
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        LOGGER.warning('未提供 token, 可能触发速率限制。建议设置 --token 或环境变量 GITHUB_TOKEN')

    ensure_dir(args.out_dir)

    with requests.Session() as session:
        session.headers.update(auth_headers(token))

        if not args.skip_list:
            LOGGER.info('开始抓取 PR 列表')
            pr_list_rows = fetch_pr_list(session, args.owner, args.repo, args.state, args.max_prs)
            write_csv(os.path.join(args.out_dir, 'pr_list.csv'), pr_list_rows)

        with open(os.path.join(args.out_dir, 'pr_list.csv'), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            pr_list_rows = [row for row in reader]

        pr_numbers = [r['number'] for r in pr_list_rows]
        print(f"Total pr_numbers: {len(pr_numbers)}")
        if not pr_numbers:
            LOGGER.info('无 PR 数据，结束。')
            return

        if not args.skip_details:
            LOGGER.info('抓取 PR 详情')
            pr_detail_rows = fetch_pr_details(session, args.owner, args.repo, pr_numbers)
            write_csv(os.path.join(args.out_dir, 'pr_details.csv'), pr_detail_rows)
        else:
            LOGGER.info('跳过 PR 详情')

        if not args.skip_files:
            LOGGER.info('抓取 PR 修改文件')
            pr_file_rows = fetch_pr_files(session, args.owner, args.repo, pr_numbers)
            write_csv(os.path.join(args.out_dir, 'pr_files.csv'), pr_file_rows)
        else:
            LOGGER.info('跳过 PR 修改文件')

        if not args.skip_reviews:
            LOGGER.info('抓取 PR 审核评论')
            pr_review_rows = fetch_pr_reviews(session, args.owner, args.repo, pr_numbers)
            write_csv(os.path.join(args.out_dir, 'pr_reviews.csv'), pr_review_rows)
        else:
            LOGGER.info('跳过 PR 审核评论')

        if not args.skip_contrib:
            LOGGER.info('抓取贡献者列表')
            contrib_rows = fetch_contributors(session, args.owner, args.repo, args.limit_contrib)
            write_csv(os.path.join(args.out_dir, 'contributors.csv'), contrib_rows)
        else:
            LOGGER.info('跳过贡献者')

    LOGGER.info('完成。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n用户中断。', file=sys.stderr)
        sys.exit(1)
