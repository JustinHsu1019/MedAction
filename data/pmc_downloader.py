# -*- coding: utf-8 -*-
"""
pmc_downloader.py
-----------------
Standalone module for searching Europe PMC and downloading PMC Open Access
full-text articles via NCBI eFetch. All Google Colab / Google Drive
dependencies have been removed; output paths are fully controlled by the caller.

Usage example:
    from pmc_downloader import search_pmc_ids, fetch_pmc_fulltext, download_pmc_dataset

    # One-liner: search + download + parse, returns list[dict]
    records = download_pmc_dataset(
        query='SRC:PMC AND OPEN_ACCESS:Y AND ("case report") AND (meningitis)',
        target_count=100,
        email="you@example.com",
        xml_save_dir="./pmc_xmls",   # optional; pass None to skip saving XMLs
    )
    # records: [{"CASE_ID": "PMC...", "RAW_TEXT": "..."}, ...]
"""

from __future__ import annotations

import os
import time
import logging
from typing import Generator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Europe PMC search → retrieve list of PMCIDs
# ---------------------------------------------------------------------------

def search_pmc_ids(
    query: str,
    target_count: int = 1000,
    page_size: int = 1000,
    sleep_between_pages: float = 0.5,
) -> list[str]:
    """
    Query the Europe PMC REST API and return up to `target_count` PMCIDs.

    Parameters
    ----------
    query : str
        Europe PMC query string (recommended to include
        ``SRC:PMC AND OPEN_ACCESS:Y``).
    target_count : int
        Maximum number of PMCIDs to retrieve.
    page_size : int
        Number of results per API request (max 1000).
    sleep_between_pages : float
        Seconds to wait between paginated requests to avoid rate limiting.

    Returns
    -------
    list[str]
        List of PMCID strings, e.g. ["PMC1234567", "PMC7654321", ...]
    """
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    ids: list[str] = []
    cursor_mark = "*"

    logger.info("Searching Europe PMC (target: %d records) ...", target_count)

    while len(ids) < target_count:
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(page_size, target_count - len(ids)),
            "cursorMark": cursor_mark,
            "resultType": "lite",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Europe PMC search failed: %s", exc)
            break

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        new_ids = [item["pmcid"] for item in results if "pmcid" in item]
        ids.extend(new_ids)
        logger.info("Retrieved %d IDs so far ...", len(ids))

        cursor_mark = data.get("nextCursorMark")
        if not cursor_mark or len(results) < page_size:
            break

        time.sleep(sleep_between_pages)

    return ids[:target_count]


# ---------------------------------------------------------------------------
# 2. XML parsing: extract RAW_TEXT from a single article XML string
# ---------------------------------------------------------------------------

def _parse_article_xml(xml_str: str, pmcid: str) -> dict | None:
    """
    Parse Title, Abstract, and Body from a single article XML string
    and concatenate them into RAW_TEXT.

    Returns
    -------
    dict | None
        {"CASE_ID": pmcid, "RAW_TEXT": ...}, or None if content is too short.
    """
    try:
        soup = BeautifulSoup(xml_str, "xml")
        parts: list[str] = []

        title = soup.find("article-title")
        if title:
            parts.append(f"Title: {title.get_text(strip=True)}")

        abstract = soup.find("abstract")
        if abstract:
            parts.append(f"Abstract: {abstract.get_text(separator=' ', strip=True)}")

        body = soup.find("body")
        if body:
            parts.append(f"Main Text: {body.get_text(separator=' ', strip=True)}")

        raw_text = "\n\n".join(parts)
        if len(raw_text) < 100:
            return None  # discard articles with insufficient content

        return {"CASE_ID": pmcid, "RAW_TEXT": raw_text}

    except Exception as exc:
        logger.warning("Failed to parse XML for %s: %s", pmcid, exc)
        return None


# ---------------------------------------------------------------------------
# 3. NCBI eFetch: batch download + parse, yielding records one by one
# ---------------------------------------------------------------------------

def fetch_pmc_fulltext(
    pmcids: list[str],
    email: str,
    batch_size: int = 50,
    sleep_between_batches: float = 1.0,
    xml_save_dir: str | None = None,
) -> Generator[dict, None, None]:
    """
    Download PMC full-text XML in batches via NCBI eFetch and yield
    parsed records one at a time.

    Parameters
    ----------
    pmcids : list[str]
        List of PMCIDs to download (accepts "PMC1234567" or bare numeric IDs).
    email : str
        Contact email required by the NCBI API usage policy.
    batch_size : int
        Number of IDs per POST request (recommended <= 50).
    sleep_between_batches : float
        Seconds to wait between batches (NCBI recommends >= 1 second).
    xml_save_dir : str | None
        Directory to save individual article XML files; pass None to skip.

    Yields
    ------
    dict
        {"CASE_ID": "PMC...", "RAW_TEXT": "..."}
    """
    if xml_save_dir:
        os.makedirs(xml_save_dir, exist_ok=True)

    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    session = requests.Session()

    total = len(pmcids)
    logger.info("Starting NCBI eFetch download: %d records, batch_size=%d", total, batch_size)

    for batch_start in range(0, total, batch_size):
        batch = pmcids[batch_start : batch_start + batch_size]

        # NCBI eFetch does not accept the "PMC" prefix in the id parameter
        id_string = ",".join(pid.replace("PMC", "") for pid in batch)

        try:
            resp = session.post(
                efetch_url,
                data={
                    "db": "pmc",
                    "id": id_string,
                    "retmode": "xml",
                    "email": email,
                },
                timeout=60,
            )
            resp.raise_for_status()

        except Exception as exc:
            logger.warning(
                "Batch %d-%d download failed: %s — skipping",
                batch_start, batch_start + len(batch), exc,
            )
            time.sleep(sleep_between_batches * 3)
            continue

        soup_batch = BeautifulSoup(resp.content, "xml")
        articles = soup_batch.find_all("article")

        for article in articles:
            # Extract PMCID from the article metadata
            id_tag = article.find("article-id", {"pub-id-type": "pmcid"})
            if not id_tag:
                continue
            raw_id = id_tag.get_text().strip()
            pmcid = raw_id if raw_id.startswith("PMC") else f"PMC{raw_id}"

            xml_str = str(article)

            # Optionally persist the raw XML
            if xml_save_dir:
                xml_path = os.path.join(xml_save_dir, f"{pmcid}.xml")
                try:
                    with open(xml_path, "w", encoding="utf-8") as fh:
                        fh.write(xml_str)
                except OSError as exc:
                    logger.warning("Could not save XML for %s: %s", pmcid, exc)

            record = _parse_article_xml(xml_str, pmcid)
            if record:
                yield record

        logger.info("Processed %d / %d records", min(batch_start + batch_size, total), total)
        time.sleep(sleep_between_batches)


# ---------------------------------------------------------------------------
# 4. All-in-one entry point: search + download + return list[dict]
# ---------------------------------------------------------------------------

def download_pmc_dataset(
    query: str,
    target_count: int,
    email: str,
    batch_size: int = 50,
    xml_save_dir: str | None = None,
    skip_pmcids: set[str] | None = None,
    sleep_search: float = 0.5,
    sleep_fetch: float = 1.0,
) -> list[dict]:
    """
    All-in-one function: search Europe PMC → download NCBI full text →
    return parsed records.

    Parameters
    ----------
    query : str
        Europe PMC query string.
    target_count : int
        Target number of records to retrieve.
    email : str
        Contact email for the NCBI API.
    batch_size : int
        Number of IDs per eFetch request (recommended <= 50).
    xml_save_dir : str | None
        Directory to save XML backups; None to skip.
    skip_pmcids : set[str] | None
        Set of already-processed PMCIDs to skip (for resuming interrupted runs).
    sleep_search : float
        Delay in seconds between search pagination requests.
    sleep_fetch : float
        Delay in seconds between eFetch batch requests.

    Returns
    -------
    list[dict]
        [{"CASE_ID": "PMC...", "RAW_TEXT": "..."}, ...]
    """
    # Step 1: retrieve candidate IDs from Europe PMC
    all_ids = search_pmc_ids(
        query=query,
        target_count=target_count,
        sleep_between_pages=sleep_search,
    )

    # Step 2: filter out already-processed IDs (resume support)
    if skip_pmcids:
        ids_to_fetch = [pid for pid in all_ids if pid not in skip_pmcids]
        logger.info(
            "Skipping %d existing IDs; %d remaining to download",
            len(all_ids) - len(ids_to_fetch), len(ids_to_fetch),
        )
    else:
        ids_to_fetch = all_ids

    # Step 3: download and parse
    records = list(
        fetch_pmc_fulltext(
            pmcids=ids_to_fetch,
            email=email,
            batch_size=batch_size,
            sleep_between_batches=sleep_fetch,
            xml_save_dir=xml_save_dir,
        )
    )

    logger.info("Done. %d valid full-text records retrieved.", len(records))
    return records
