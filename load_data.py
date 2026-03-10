"""
load_data.py
Initializes the SQLite database schema and loads data from cell-count.csv.
Run: python load_data.py
"""

import sqlite3
import csv
import os

DB_PATH = "immune_trial.db"
CSV_PATH = "cell-count.csv"


def init_db(conn):
    """Create normalized relational schema."""
    cur = conn.cursor()

    # subjects: one row per unique subject with demographic/trial metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id   TEXT PRIMARY KEY,
            project      TEXT NOT NULL,
            condition    TEXT,
            age          INTEGER,
            sex          TEXT,
            treatment    TEXT,
            response     TEXT
        )
    """)

    # samples: one row per sample, foreign key to subjects
    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id                  TEXT PRIMARY KEY,
            subject_id                 TEXT NOT NULL,
            sample_type                TEXT,
            time_from_treatment_start  INTEGER,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        )
    """)

    # cell_counts: one row per (sample, population) — long / tidy format
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cell_counts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id   TEXT NOT NULL,
            population  TEXT NOT NULL,
            count       INTEGER NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
        )
    """)

    # Indexes for common query patterns
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cc_sample   ON cell_counts(sample_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cc_pop      ON cell_counts(population)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_smp_subject ON samples(subject_id)")

    conn.commit()


POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


def load_csv(conn, csv_path):
    """Load cell-count.csv into the normalized schema (idempotent)."""
    cur = conn.cursor()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Upsert subject
            cur.execute("""
                INSERT OR IGNORE INTO subjects
                    (subject_id, project, condition, age, sex, treatment, response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["subject"],
                row["project"],
                row["condition"],
                int(row["age"]),
                row["sex"],
                row["treatment"],
                row["response"],
            ))

            # Upsert sample
            cur.execute("""
                INSERT OR IGNORE INTO samples
                    (sample_id, subject_id, sample_type, time_from_treatment_start)
                VALUES (?, ?, ?, ?)
            """, (
                row["sample"],
                row["subject"],
                row["sample_type"],
                int(row["time_from_treatment_start"]),
            ))

            # Insert cell counts (skip if already loaded)
            existing = cur.execute(
                "SELECT COUNT(*) FROM cell_counts WHERE sample_id = ?", (row["sample"],)
            ).fetchone()[0]

            if existing == 0:
                for pop in POPULATIONS:
                    cur.execute("""
                        INSERT INTO cell_counts (sample_id, population, count)
                        VALUES (?, ?, ?)
                    """, (row["sample"], pop, int(row[pop])))

    conn.commit()
    print(f"Data loaded successfully into '{DB_PATH}'.")


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh load each run

    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        load_csv(conn, CSV_PATH)
        # Quick sanity check
        n_subjects = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        n_samples  = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n_counts   = conn.execute("SELECT COUNT(*) FROM cell_counts").fetchone()[0]
        print(f"  Subjects : {n_subjects}")
        print(f"  Samples  : {n_samples}")
        print(f"  Cell rows: {n_counts}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
