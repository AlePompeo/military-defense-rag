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
from contextualizer import Contextualizer
from embedder import Embedder, Reranker
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

# Human-readable titles, prepended to each chunk before embedding/indexing
# (cheap "contextual retrieval": disambiguates generic-sounding passages —
# e.g. a paragraph on "friction" reads very differently from Clausewitz vs.
# a modern FM — without the cost of an LLM call per chunk at ingest time).
TITLES: dict[str, str] = {
    "dark_future_asymmetric_warfighting": "The Dark Future of Next-Gen Asymmetric Warfighting",
    "cia_gateway_process": "CIA Analysis and Assessment of Gateway Process",
    "mcdp1_warfighting": "MCDP 1 Warfighting (USMC)",
    "on_war_clausewitz": "On War (Clausewitz)",
    "art_of_war_sun_tzu": "The Art of War (Sun Tzu)",
    "fm_3_0": "FM 3-0 Operations",
    "fm_5_0": "FM 5-0 Planning and Orders Production",
    "atp_2_01_3_ipb": "ATP 2-01.3 Intelligence Preparation of the Battlefield",
    "fm_2_0": "FM 2-0 Intelligence",
    "atp_3_60_targeting": "ATP 3-60 Targeting",
    "fm_3_06_urban": "FM 3-06 Urban Operations",
    "fm_3_12_cyber_ew": "FM 3-12 Cyberspace and Electromagnetic Warfare",
    "adp_3_0": "ADP 3-0 Operations",
    "operations_research": "Operations Research (2nd Edition)",
    "data_structures_algorithms": "Data Structures and Algorithms",
    "pathfinding_nav_mesh": "Pathfinding Algorithms in Navigational Meshes",
    "game_theory": "Game Theory",
    "fm_3_09_fire_support": "FM 3-09 Fire Support and Field Artillery Operations",
    "fm_3_24_insurgencies": "FM 3-24 Insurgencies and Countering Insurgencies",
    "ajp_3_3_air_space": "AJP-3.3 Allied Joint Doctrine for Air and Space Operations",
    "ajp_3_20_cyberspace": "AJP-3.20 Allied Joint Doctrine for Cyberspace Operations",
    "lanchester_warfare_models": "Lanchester-Type Models of Warfare",
    "military_ops_research_1994": "Military Operations Research (Summer 1994)",
    "math_models_mil_logistics": "Mathematical Models for Planning in Military and Humanitarian Logistics",
    "gis_sim_military_path": "GIS-Based Simulation Model for Military Path Planning of Unmanned Ground Robots",
    "military_route_planning": "Military Route Planning in Battlefield Simulation",
}


def _compose_text(doc: dict, title: str | None, contextualizer: Contextualizer | None) -> str:
    """Final indexed/embedded text = title, then optional LLM-generated
    context (opt-in, `--contextualize`), then the clean child chunk."""
    parts = []
    if title:
        parts.append(title)
    if contextualizer is not None:
        parts.append(contextualizer.contextualize(doc["parent_text"], doc["text"]))
    parts.append(doc["text"])
    return "\n\n".join(parts)


def cmd_ingest(store: VectorStore, contextualizer: Contextualizer | None) -> None:
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
            docs = load_document(
                url, name, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.PARENT_CHUNK_SIZE, **opts
            )
            title = TITLES.get(name)
            # Contextualizing is one LLM call per chunk — slow enough to need
            # its own progress bar rather than looking frozen for a whole document.
            chunk_iter = (
                tqdm(docs, unit="chunk", leave=False, desc=f"  {name}") if contextualizer else docs
            )
            for doc in chunk_iter:
                doc["text"] = _compose_text(doc, title, contextualizer)
            store.add(docs)
            tqdm.write(f"  {name}: {len(docs)} chunks")
        except Exception as exc:
            tqdm.write(f"  {name}: FAILED — {exc}")
            failed.append(name)

    print(f"\nDone. Total chunks in store: {store.count()}")
    if failed:
        print(f"Failed sources ({len(failed)}): {', '.join(failed)}")
        print("Re-run `python main.py ingest` to retry failed sources.")


def cmd_query(store: VectorStore, question: str | None, use_reranker: bool, use_hyde: bool) -> None:
    rag = RAG(
        store,
        config.LM_STUDIO_URL,
        config.LM_STUDIO_MODEL,
        config.TOP_K,
        config.RETRIEVAL_CANDIDATES,
        use_reranker,
        config.MMR_LAMBDA,
        use_hyde,
        config.HYDE_CACHE_PATH,
        config.GENERATION_CACHE_PATH,
    )

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

    i_parser = sub.add_parser("ingest", help="Download and index all documents")
    i_parser.add_argument(
        "--contextualize", action="store_true",
        help=(
            "Generate an LLM contextual summary per chunk before embedding (Anthropic's "
            "'Contextual Retrieval', adapted to parent-block context). One local LLM call "
            "per chunk: impractical CPU-only (multi-day for this corpus), reasonable on a "
            "dedicated GPU (roughly hours). Off by default."
        ),
    )

    q_parser = sub.add_parser("query", help="Query the RAG")
    q_parser.add_argument("-q", "--question", help="One-shot question (omit for interactive mode)")
    q_parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip cross-encoder reranking (faster, less precise; useful on constrained RAM)",
    )
    q_parser.add_argument(
        "--hyde", action="store_true",
        help=(
            "HyDE: generate a hypothetical answer passage via the local LLM and use it to "
            "steer dense retrieval, instead of the raw question. One extra LLM call per "
            "(uncached) query — added latency on top of generation itself."
        ),
    )

    args = parser.parse_args()

    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    embedder = Embedder(config.EMBEDDING_MODEL, config.EMBEDDING_CACHE_PATH)
    reranker = Reranker(config.RERANKER_MODEL)
    store = VectorStore(
        config.CHROMA_DB_PATH,
        embedder,
        reranker,
        query_cache_dir=config.QUERY_CACHE_PATH,
        query_cache_ttl=config.QUERY_CACHE_TTL,
        hnsw_ef_construction=config.HNSW_EF_CONSTRUCTION,
        hnsw_ef_search=config.HNSW_EF_SEARCH,
        hnsw_m=config.HNSW_M,
    )

    if args.command == "ingest":
        contextualizer = (
            Contextualizer(config.LM_STUDIO_URL, config.LM_STUDIO_MODEL, config.CONTEXTUALIZE_CACHE_PATH)
            if args.contextualize
            else None
        )
        cmd_ingest(store, contextualizer)
    elif args.command == "query":
        use_reranker = config.USE_RERANKER and not args.no_rerank
        use_hyde = config.USE_HYDE or args.hyde
        cmd_query(store, getattr(args, "question", None), use_reranker, use_hyde)


if __name__ == "__main__":
    main()
