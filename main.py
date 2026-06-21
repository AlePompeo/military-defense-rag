"""
CLI entry point.

  python main.py ingest          # download and index all 26 documents
  python main.py query           # interactive REPL
  python main.py query -q "..."  # one-shot question
"""

import argparse
import os
import sys
from tqdm import tqdm

import config
from loader import load_document
from store import VectorStore
from rag import RAG


DOCUMENTS: dict[str, str | tuple[str, dict]] = {
    "dark_future_asymmetric_warfighting": "https://arxiv.org/pdf/2408.12045",
    # Scanned PDF — browser UA triggers a cia.gov redirect loop; plain UA serves it directly.
    # OCR (pytesseract + pdf2image) is used automatically for the image-only pages.
    "cia_gateway_process": (
        "https://www.cia.gov/readingroom/docs/CIA-RDP96-00788R001700210016-5.pdf",
        {"headers": {"User-Agent": "Mozilla/5.0 (RAG research client)"}},
    ),
    "mcdp1_warfighting": "https://upload.wikimedia.org/wikipedia/commons/4/4e/MCDP_1_Warfighting.pdf",
    # Gutenberg geo-blocked in Italy; using fulltextarchive.com and tantor S3 as alternatives.
    "on_war_clausewitz": "https://icct.nl/sites/default/files/import/publication/On-War.pdf",
    "art_of_war_sun_tzu": "https://tantor-site-assets.s3.amazonaws.com/bonus-content/0841_ArtWarRevised/0841_ArtWarRevised_ebook.pdf",
    "fm_3_0": "https://soldat-und-technik.de/wp-content/uploads/2022/10/ARN36290-FM_3-0.pdf",
    "fm_5_0": "https://stephengates.com/ADM/FM-JUL22.pdf",
    "atp_2_01_3_ipb": "https://home.army.mil/wood/application/files/8915/5751/8365/ATP_2-01.3_Intelligence_Preparation_of_the_Battlefield.pdf",
    "fm_2_0": "https://www.bits.de/NRANEU/others/amd-us-archive/fm2-0fd%2809%29.pdf",
    # irp.fas.org blocks bots; replaced with verified working mirrors.
    "atp_3_60_targeting": "https://www.utahmilitia.org/members/lib/adp/atp3-60.pdf",
    "fm_3_06_urban": "https://www.1215.org/lawnotes/misc/army-field-manual-fm3-06-urban-operations.pdf",
    "fm_3_12_cyber_ew": "https://nsarchive.gwu.edu/sites/default/files/documents/3678217/Document-11-Department-of-the-Army-FM-3-12.pdf",
    "adp_3_0": "https://www.bits.de/NRANEU/others/amd-us-archive/ADP3-0%2816%29.pdf",
    "operations_research": "https://www.bbau.ac.in/dept/UIET/EME-601%20Operation%20Research.pdf",
    "data_structures_algorithms": "https://mta.ca/~rrosebru/oldcourse/263114/Dsa.pdf",
    "pathfinding_nav_mesh": "https://www.cs.csustan.edu/~mmartin/teaching/CS4960S15/Corey%20Trevena%20-%20Pathfinding%20Algorithms%20in%20Navigational%20Meshes%20PDF.pdf",
    "game_theory": "https://didattica.unibocconi.it/mypage/upload/48808_20220802_072515_02.08.2022TEXTBOOKGTAST_COMPRESSED.PDF",
    "fm_3_09_fire_support": "https://www.revista-artilharia.pt/admin/upload/ficheiros/ficheirosMultimedia/fm-3-09-fire-support-and-field-artillery-operations.pdf",
    "fm_3_24_insurgencies": "https://www.globalsecurity.org/military/library/policy/army/fm/3-24/fm3-24.pdf",
    "ajp_3_3_air_space": "https://www.coemed.org/files/stanags/01_AJP/AJP-3.3_EDB_V1_E_3700.pdf",
    "ajp_3_20_cyberspace": "https://assets.publishing.service.gov.uk/media/5f086ec4d3bf7f2bef137675/doctrine_nato_cyberspace_operations_ajp_3_20_1_.pdf",
    "lanchester_warfare_models": "https://apps.dtic.mil/sti/tr/pdf/ADA090842.pdf",
    "military_ops_research_1994": "https://apps.dtic.mil/sti/tr/pdf/ADA321335.pdf",
    "math_models_mil_logistics": "https://publications.tno.nl/publication/34644360/brDxfOYa/wagenvoort-2025-mathematical.pdf",
    "gis_sim_military_path": "https://www.witpress.com/Secure/ejournals/papers/SSE010302f.pdf",
    "military_route_planning": "http://www.tarapata.strefa.pl/publikacje/jtit_2003.pdf",
}


def cmd_ingest(store: VectorStore) -> None:
    already = set(store.indexed_sources())
    pending = {k: v for k, v in DOCUMENTS.items() if k not in already}

    if not pending:
        print(f"All {len(DOCUMENTS)} sources already indexed ({store.count()} chunks total).")
        return

    if already:
        print(f"Skipping {len(already)} already-indexed sources.")

    print(f"Ingesting {len(pending)} source(s)...\n")
    failed: list[str] = []

    for name, entry in tqdm(pending.items(), unit="doc"):
        url, opts = (entry, {}) if isinstance(entry, str) else entry
        try:
            docs = load_document(url, name, config.CHUNK_SIZE, config.CHUNK_OVERLAP, **opts)
            store.add(docs)
            tqdm.write(f"  {name}: {len(docs)} chunks")
        except Exception as exc:
            tqdm.write(f"  {name}: FAILED — {exc}")
            failed.append(name)

    print(f"\nDone. Total chunks in store: {store.count()}")
    if failed:
        print(f"Failed sources ({len(failed)}): {', '.join(failed)}")
        print("Re-run `python main.py ingest` to retry failed sources.")


def cmd_query(store: VectorStore, question: str | None) -> None:
    rag = RAG(store, config.LM_STUDIO_URL, config.LM_STUDIO_MODEL, config.TOP_K)

    if question:
        print(rag.query(question))
        return

    print("Military & Defense RAG  |  type 'quit' to exit\n")
    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        print()
        print(rag.query(q))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Military & Defense RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Download and index all documents")

    q_parser = sub.add_parser("query", help="Query the RAG")
    q_parser.add_argument("-q", "--question", help="One-shot question (omit for interactive mode)")

    args = parser.parse_args()

    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    store = VectorStore(config.CHROMA_DB_PATH)

    if args.command == "ingest":
        cmd_ingest(store)
    elif args.command == "query":
        cmd_query(store, getattr(args, "question", None))


if __name__ == "__main__":
    main()
