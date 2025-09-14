from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


# ----------------- helpers -----------------
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
	"""Read CSV if exists and non-empty; else return None."""
	if not path.exists() or path.stat().st_size == 0:
		return None
	try:
		return pd.read_csv(path)
	except Exception:
		# Fallback encodings / engine can be added here if needed
		return pd.read_csv(path, engine="python", encoding_errors="ignore")


def _word_count(text: str) -> int:
	if not isinstance(text, str) or not text:
		return 0
	return len(re.findall(r"\b[A-Za-z0-9_]+\b", text))


def _contains_any(text: str, patterns: list[str]) -> int:
	if not isinstance(text, str) or not text:
		return 0
	text_l = text.lower()
	for p in patterns:
		if re.search(rf"\b{re.escape(p)}\b", text_l):
			return 1
	return 0


def _infer_pr_key(df: pd.DataFrame) -> Optional[str]:
	"""Best-effort PR key inference for joins across CSVs."""
	for cand in ("pr_number", "number", "pull_number", "pr id", "pr_id", "issue_number"):
		if cand in df.columns:
			return cand
	return None


def _to_datetime(s: pd.Series) -> pd.Series:
	return pd.to_datetime(s, errors="coerce")


# ----------------- reviews aggregation -----------------
def _aggregate_reviews(repo_dir: Path) -> Optional[pd.DataFrame]:
	reviews_csv = repo_dir / "pr_reviews.csv"
	df = _safe_read_csv(reviews_csv)
	if df is None or df.empty:
		return None

	pr_key = _infer_pr_key(df)
	if pr_key is None:
		return None

	# Normalize columns that may be present
	if "submitted_at" in df.columns:
		df["submitted_at"] = _to_datetime(df["submitted_at"])
	else:
		df["submitted_at"] = pd.NaT

	if "body" not in df.columns:
		df["body"] = ""

	if "user/login" not in df.columns:
		df["user/login"] = df.get("user", df.get("login", "")).fillna("")

	# Basic per-review measures
	df["body_words"] = df["body"].fillna("").astype(str).map(_word_count)

	# Per-PR aggregates
	gb = df.groupby(pr_key, dropna=False)
	agg = pd.DataFrame({
		pr_key: gb.size().index,
		"reviews_count": gb.size().values,
		"reviews_users": gb["user/login"].nunique().values,
		"reviews_avg_len": gb["body_words"].mean().fillna(0).values,
	})

	# Time span (in hours) between first and last review submission
	time_span = (
		gb["submitted_at"].agg(lambda s: (s.max() - s.min()).total_seconds() / 3600.0 if s.notna().any() else 0.0)
		.rename("reviews_time_span_hours")
	)
	agg = agg.merge(time_span, left_on=pr_key, right_index=True, how="left")
	return agg


# ----------------- files aggregation -----------------
def _aggregate_files(repo_dir: Path) -> Optional[pd.DataFrame]:
	files_csv = repo_dir / "pr_files.csv"
	df = _safe_read_csv(files_csv)
	if df is None or df.empty:
		return None

	pr_key = _infer_pr_key(df)
	if pr_key is None:
		return None

	# Normalize expected columns
	for col in ("additions", "deletions", "changes"):
		if col not in df.columns:
			df[col] = 0
		df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

	# Patch length stats (characters and lines)
	if "patch" in df.columns:
		patch_str = df["patch"].fillna("").astype(str)
		df["patch_len_chars"] = patch_str.str.len()
		df["patch_len_lines"] = patch_str.map(lambda s: s.count("\n") + (1 if s else 0))
	else:
		df["patch_len_chars"] = 0
		df["patch_len_lines"] = 0

	gb = df.groupby(pr_key, dropna=False)
	agg = pd.DataFrame({
		pr_key: gb.size().index,
		"files_count": gb.size().values,
		"files_avg_additions": gb["additions"].mean().fillna(0).values,
		"files_avg_deletions": gb["deletions"].mean().fillna(0).values,
		"files_avg_changes": gb["changes"].mean().fillna(0).values,
		"files_avg_patch_chars": gb["patch_len_chars"].mean().fillna(0).values,
		"files_avg_patch_lines": gb["patch_len_lines"].mean().fillna(0).values,
	})
	return agg


# ----------------- core details-based features -----------------
def _extract_details_features(details_csv: Path) -> pd.DataFrame:
	if not details_csv.exists():
		# No details.csv -> return empty DataFrame; caller decides how to proceed.
		return pd.DataFrame()

	df = _safe_read_csv(details_csv)
	if df is None or df.empty:
		return pd.DataFrame()

	# Ensure required columns exist
	for col in ["title", "body", "user/login", "created_at"]:
		if col not in df.columns:
			df[col] = "" if col in ("title", "body", "user/login") else pd.NaT

	df["title"] = df["title"].fillna("").astype(str)
	df["body"] = df["body"].fillna("").astype(str)
	df["user/login"] = df["user/login"].fillna("").astype(str)
	df["created_at"] = _to_datetime(df["created_at"])

	# Numeric fields
	for col in ["additions", "deletions", "changed_files", "commits", "number"]:
		if col not in df.columns:
			df[col] = 0
		df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int, errors="ignore")

	# Textual flags
	title_body = (df["title"] + " " + df["body"]).fillna("")
	has_test = title_body.map(lambda s: _contains_any(s, ["test"]))
	has_bug = title_body.map(lambda s: _contains_any(s, ["bug"]))
	has_feature = title_body.map(lambda s: _contains_any(s, ["feature"]))
	has_improve = title_body.map(lambda s: _contains_any(s, ["improve", "improvement"]))
	has_document = title_body.map(lambda s: _contains_any(s, ["doc", "docs", "documentation"]))
	has_refactor = title_body.map(lambda s: _contains_any(s, ["refactor"]))

	# Author history features
	df = df.sort_values(["user/login", "created_at"], kind="mergesort")
	prev_prs = df.groupby("user/login", dropna=False).cumcount().astype(int)
	change_num = df.groupby("user/login", dropna=False)["commits"].transform("mean").fillna(0)

	# Length features
	title_words = df["title"].map(_word_count)
	body_words = df["body"].map(_word_count)

	out = pd.DataFrame({
		"number": df["number"] if "number" in df.columns else pd.Series(range(len(df))),
		"has_test": has_test.astype(int),
		"has_bug": has_bug.astype(int),
		"has_feature": has_feature.astype(int),
		"has_improve": has_improve.astype(int),
		"has_document": has_document.astype(int),
		"has_refactor": has_refactor.astype(int),
		"lines_added": pd.to_numeric(df["additions"], errors="coerce").fillna(0).astype(int, errors="ignore"),
		"lines_deleted": pd.to_numeric(df["deletions"], errors="coerce").fillna(0).astype(int, errors="ignore"),
		"files_changed": pd.to_numeric(df["changed_files"], errors="coerce").fillna(0).astype(int, errors="ignore"),
		"commits": pd.to_numeric(df["commits"], errors="coerce").fillna(0).astype(int, errors="ignore"),
		"change_num": pd.to_numeric(change_num, errors="coerce").fillna(0),
		"prev_PRs": prev_prs,
		"title_words": title_words,
		"body_words": body_words,
	})
	return out


# ----------------- main orchestration -----------------
def extract_features_for_repo(repo_dir: Path) -> pd.DataFrame:
	repo_name = repo_dir.name
	details_csv = repo_dir / "pr_details.csv"

	# Base (details) features
	base = _extract_details_features(details_csv)

	# If details missing, we still try to build from reviews/files but need a PR key named 'number'
	if base.empty:
		base = pd.DataFrame({"number": []})

	# Reviews aggregates
	review_agg = _aggregate_reviews(repo_dir)
	if review_agg is not None and not review_agg.empty:
		pr_key = _infer_pr_key(review_agg)
		if pr_key and pr_key != "number":
			review_agg = review_agg.rename(columns={pr_key: "number"})
		base = base.merge(review_agg, on="number", how="left")

	# Files aggregates
	files_agg = _aggregate_files(repo_dir)
	if files_agg is not None and not files_agg.empty:
		pr_key = _infer_pr_key(files_agg)
		if pr_key and pr_key != "number":
			files_agg = files_agg.rename(columns={pr_key: "number"})
		base = base.merge(files_agg, on="number", how="left")

	# Fill NaNs for any newly added numeric columns
	for col in base.columns:
		if pd.api.types.is_numeric_dtype(base[col]):
			base[col] = base[col].fillna(0)

	# Save
	out_path = repo_dir / f"{repo_name}_features.csv"
	base.to_csv(out_path, index=False)
	return base


def main():
	base = Path(__file__).resolve().parent
	repo_dir = base / "llvm"
	extract_features_for_repo(repo_dir)


if __name__ == "__main__":
	main()
