import sqlite3
import pandas as pd

conn = sqlite3.connect("datos_teleco/teleco_pricing.db")

query = """
SELECT
    c.segmento,
    con.plan_contratado,
    COUNT(*)                                    AS contratos,
    ROUND(AVG(con.tarifa_contratada), 2)        AS tarifa_media,
    ROUND(MIN(con.tarifa_contratada), 2)        AS tarifa_min,
    ROUND(MAX(con.tarifa_contratada), 2)        AS tarifa_max,
    ROUND(AVG(con.descuento_aplicado) * 100, 1) AS descuento_medio_pct,
    ROUND(AVG(r.margen_bruto_pct), 1)           AS margen_medio_pct
FROM contratos con
JOIN clientes c ON con.cliente_id = c.cliente_id
JOIN red_costes r ON con.contrato_id = r.contrato_id
GROUP BY c.segmento, con.plan_contratado
ORDER BY c.segmento, tarifa_media
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(df.to_string())