import pandas as pd
from pathlib import Path

BASE_DIR = Path(".")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

TEST_CSV = BASE_DIR / "test_data.csv"
Q3_CSV = CHECKPOINTS_DIR / "q3" / "_q3.csv"
Q4_CSV = CHECKPOINTS_DIR / "q4" / "_q4.csv"
Q5_CSV = CHECKPOINTS_DIR / "q5" / "_q5.csv"
Q6_CSV = CHECKPOINTS_DIR / "q6" / "_q6.csv"

OUTPUT_CSV = BASE_DIR / "submission.csv"


def main() -> None:
    base = pd.read_csv(TEST_CSV)
    q3 = pd.read_csv(Q3_CSV)
    q4 = pd.read_csv(Q4_CSV)
    q5 = pd.read_csv(Q5_CSV)
    q6 = pd.read_csv(Q6_CSV)

    base = base.set_index("id")
    q3 = q3.set_index("id")
    q4 = q4.set_index("id")
    q5 = q5.set_index("id")
    q6 = q6.set_index("id")

    # Fill existing columns in test_data.csv
    base["Honesty-Humility"] = q3["H_self"].reindex(base.index)
    base["Extraversion"] = q4["E_self"].reindex(base.index)
    base["Agreeableness"] = q5["A_self"].reindex(base.index)
    base["Conscientiousness"] = q6["C_self"].reindex(base.index)

    submission = base[
        ["Honesty-Humility", "Extraversion", "Agreeableness", "Conscientiousness"]
    ].reset_index()
    submission.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
