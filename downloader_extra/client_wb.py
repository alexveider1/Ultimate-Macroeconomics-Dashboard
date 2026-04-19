import wbgapi as wb
import polars as pl
from sqlalchemy import create_engine, text


def fetch_and_store_indicator(indicator_id: str, wb_db_id: int, sql_uri: str) -> int:
    records = wb.data.fetch(
        indicator_id,
        db=wb_db_id,
        skipAggs=True,
        economy="all",
        time="all",
        skipBlanks=False,
        numericTimeKeys=True,
    )

    rows = list(records)
    if not rows:
        raise ValueError(f"No data found for indicator id: {indicator_id}")

    df = pl.DataFrame(rows)
    economy_column = "economy"
    year_column = "date"

    df_transformed = df.select(
        [
            pl.col(economy_column).alias("economy"),
            pl.col(year_column).alias("year"),
            pl.col("value"),
        ]
    ).with_columns(
        [
            pl.lit(indicator_id).alias("indicator_id"),
            pl.lit(wb_db_id).alias("db_id"),
        ]
    )

    df_transformed = df_transformed.drop_nulls(subset=["economy", "year"])

    df_transformed = df_transformed.with_columns(
        [
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("value").cast(pl.Float64, strict=False),
        ]
    )

    if df_transformed.is_empty():
        raise ValueError(
            f"No non-null rows found for indicator id: {indicator_id} in db: {wb_db_id}"
        )

    df_transformed = df_transformed.unique(
        subset=["economy", "year", "indicator_id", "db_id"],
        keep="last",
        maintain_order=True,
    )

    rows_to_insert = df_transformed.to_dicts()
    engine = create_engine(sql_uri)
    delete_statement = text(
        """
        DELETE FROM indicators
        WHERE indicator_id = :indicator_id AND db_id = :db_id
        """
    )
    insert_statement = text(
        """
        INSERT INTO indicators (economy, year, value, indicator_id, db_id)
        VALUES (:economy, :year, :value, :indicator_id, :db_id)
        """
    )

    try:
        with engine.begin() as connection:
            connection.execute(
                delete_statement,
                {"indicator_id": indicator_id, "db_id": wb_db_id},
            )
            connection.execute(insert_statement, rows_to_insert)
    finally:
        engine.dispose()

    return len(rows_to_insert)
