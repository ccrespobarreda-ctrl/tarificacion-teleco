"""
Dataset Sintético — Tarificación Dinámica Telecomunicaciones
=============================================================
Simula un operador de telecomunicaciones estilo Movistar/Vodafone España

Tablas generadas:
  1. clientes              (~95.000 clientes)
  2. contratos             (~95.000 contratos de datos móviles)
  3. consumo_mensual       (~1.140.000 registros — 12 meses por contrato)
  4. red_costes            (~95.000 registros de coste de red por contrato)
  5. competencia_mercado   (~95.000 registros de precios por segmento

Variable objetivo: tarifa_contratada (€/mes)
"""

import sqlite3
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
SEED       = 42
N_CLIENTES = 95_000
OUTPUT_DIR = "datos_teleco"

random.seed(SEED)
np.random.seed(SEED)
fake = Faker("es_ES")
Faker.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("⚙️  Iniciando generación del dataset de telecomunicaciones...\n")


# ═══════════════════════════════════════════════
# TABLA 1 — clientes
# ═══════════════════════════════════════════════
print("📋 Generando tabla: clientes...")

SEGMENTOS = ["Residencial", "PYME", "Corporativo"]
PESOS_SEG = [0.65, 0.25, 0.10]

CCAA = [
    "Madrid", "Cataluña", "Andalucía", "Valenciana", "País Vasco",
    "Galicia", "Castilla y León", "Aragón", "Canarias", "Murcia"
]
PESOS_CCAA = [0.22, 0.19, 0.15, 0.11, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

CANALES = ["Online", "Tienda física", "Teléfono", "Distribuidor"]
PESOS_CANAL = [0.40, 0.30, 0.20, 0.10]

segmentos  = np.random.choice(SEGMENTOS, N_CLIENTES, p=PESOS_SEG)
edades     = np.where(
    segmentos == "Residencial", np.random.normal(38, 12, N_CLIENTES).clip(18, 75),
    np.where(segmentos == "PYME", np.random.normal(45, 10, N_CLIENTES).clip(25, 70),
             np.random.normal(42, 8, N_CLIENTES).clip(28, 65))
).astype(int)

ingresos = np.where(
    segmentos == "Residencial", np.random.normal(1800, 500, N_CLIENTES).clip(800, 6000),
    np.where(segmentos == "PYME", np.random.normal(3500, 1000, N_CLIENTES).clip(1500, 15000),
             np.random.normal(8000, 2000, N_CLIENTES).clip(3000, 30000))
).round(2)

antiguedad = np.random.exponential(scale=36, size=N_CLIENTES).clip(1, 180).astype(int)

clientes = pd.DataFrame({
    "cliente_id"           : [f"CLI{str(i).zfill(6)}" for i in range(1, N_CLIENTES + 1)],
    "nombre"               : [fake.first_name() for _ in range(N_CLIENTES)],
    "apellidos"            : [fake.last_name() + " " + fake.last_name() for _ in range(N_CLIENTES)],
    "edad"                 : edades,
    "segmento"             : segmentos,
    "comunidad_autonoma"   : np.random.choice(CCAA, N_CLIENTES, p=PESOS_CCAA),
    "canal_adquisicion"    : np.random.choice(CANALES, N_CLIENTES, p=PESOS_CANAL),
    "ingreso_estimado"     : ingresos,
    "antiguedad_meses"     : antiguedad,
    "num_lineas_activas"   : np.where(segmentos == "Residencial",
                                np.random.randint(1, 5, N_CLIENTES),
                                np.where(segmentos == "PYME",
                                    np.random.randint(3, 20, N_CLIENTES),
                                    np.random.randint(10, 100, N_CLIENTES))),
    "tiene_fibra"          : np.random.choice([0, 1], N_CLIENTES, p=[0.45, 0.55]),
    "tiene_tv"             : np.random.choice([0, 1], N_CLIENTES, p=[0.60, 0.40]),
    "nps_score"            : np.random.randint(0, 11, N_CLIENTES),
    "fecha_alta"           : [
        (datetime(2024, 3, 28) - timedelta(days=int(a * 30))).strftime("%Y-%m-%d")
        for a in antiguedad
    ],
})

print(f"   ✓ {len(clientes):,} clientes generados")


# ═══════════════════════════════════════════════
# TABLA 2 — contratos
# ═══════════════════════════════════════════════
print("📋 Generando tabla: contratos...")

PLANES = {
    "Residencial": ["Básico 5GB", "Estándar 20GB", "Premium 50GB", "Ilimitado"],
    "PYME"       : ["PYME Básico 20GB", "PYME Pro 50GB", "PYME Total Ilimitado"],
    "Corporativo": ["Corp Standard", "Corp Advanced", "Corp Enterprise"]
}

TARIFAS_BASE = {
    "Básico 5GB"          : (8,  15),
    "Estándar 20GB"       : (15, 25),
    "Premium 50GB"        : (22, 35),
    "Ilimitado"           : (30, 50),
    "PYME Básico 20GB"    : (20, 35),
    "PYME Pro 50GB"       : (30, 50),
    "PYME Total Ilimitado": (40, 70),
    "Corp Standard"       : (25, 45),
    "Corp Advanced"       : (40, 70),
    "Corp Enterprise"     : (60, 120),
}

planes_lista       = []
tarifas_lista      = []
descuentos_lista   = []
permanencias_lista = []

for seg in segmentos:
    plan = random.choice(PLANES[seg])
    rango = TARIFAS_BASE[plan]

    # Tarifa base con variación por negociación
    tarifa_base = round(random.uniform(rango[0], rango[1]), 2)

    # Descuento por fidelización, multiproducto, promoción
    descuento = 0
    if random.random() < 0.4:   # 40% tiene descuento
        descuento = round(random.uniform(0.05, 0.25), 2)

    tarifa_final = round(tarifa_base * (1 - descuento), 2)
    permanencia  = random.choice([0, 12, 18, 24]) if seg == "Residencial" else random.choice([12, 24, 36])

    planes_lista.append(plan)
    tarifas_lista.append(tarifa_final)
    descuentos_lista.append(descuento)
    permanencias_lista.append(permanencia)

datos_contratados_gb = []
for plan in planes_lista:
    if "5GB" in plan:       datos_contratados_gb.append(5)
    elif "20GB" in plan:    datos_contratados_gb.append(20)
    elif "50GB" in plan:    datos_contratados_gb.append(50)
    else:                   datos_contratados_gb.append(999)  # ilimitado

contratos = pd.DataFrame({
    "contrato_id"          : [f"CON{str(i).zfill(6)}" for i in range(1, N_CLIENTES + 1)],
    "cliente_id"           : clientes["cliente_id"],
    "plan_contratado"      : planes_lista,
    "datos_contratados_gb" : datos_contratados_gb,
    "tarifa_contratada"    : tarifas_lista,       # ← VARIABLE OBJETIVO
    "descuento_aplicado"   : descuentos_lista,
    "permanencia_meses"    : permanencias_lista,
    "roaming_activo"       : np.random.choice([0, 1], N_CLIENTES, p=[0.70, 0.30]),
    "seguro_dispositivo"   : np.random.choice([0, 1], N_CLIENTES, p=[0.75, 0.25]),
    "fecha_inicio_contrato": [
        (datetime(2024, 3, 28) - timedelta(days=random.randint(30, 1095))).strftime("%Y-%m-%d")
        for _ in range(N_CLIENTES)
    ],
    "estado_contrato"      : np.random.choice(
        ["Activo", "Activo", "Activo", "Suspendido", "Baja"], N_CLIENTES, p=[0.85, 0.05, 0.04, 0.04, 0.02]
    ),
})

print(f"   ✓ {len(contratos):,} contratos generados")
print(f"   ✓ Tarifa media: {contratos['tarifa_contratada'].mean():.2f}€")


# ═══════════════════════════════════════════════
# TABLA 3 — consumo_mensual (12 meses por contrato)
# ═══════════════════════════════════════════════
print("📋 Generando tabla: consumo_mensual (esto tarda ~60 seg)...")

consumo_rows = []
for i, (_, con) in enumerate(contratos.iterrows()):
    limite_gb = con["datos_contratados_gb"]
    for mes in range(1, 13):
        # Consumo realista según plan
        if limite_gb == 999:  # ilimitado
            consumo_gb = round(random.gauss(35, 12), 2)
        else:
            pct_uso = random.gauss(0.75, 0.20)
            consumo_gb = round(max(0.1, min(pct_uso, 1.3) * limite_gb), 2)

        # Llamadas y SMS
        minutos_llamadas = max(0, int(random.gauss(180, 80)))
        num_sms          = max(0, int(random.gauss(20, 15)))
        roaming_gb       = round(max(0, random.gauss(0.5, 1.0)), 2) if con["roaming_activo"] else 0

        # Excesos de datos
        exceso_datos_gb  = max(0, round(consumo_gb - limite_gb, 2)) if limite_gb != 999 else 0
        cargo_exceso     = round(exceso_datos_gb * 3.5, 2)  # 3.5€/GB de exceso

        consumo_rows.append({
            "consumo_id"       : None,
            "contrato_id"      : con["contrato_id"],
            "cliente_id"       : con["cliente_id"],
            "mes"              : mes,
            "consumo_datos_gb" : consumo_gb,
            "minutos_llamadas" : minutos_llamadas,
            "num_sms"          : num_sms,
            "roaming_gb"       : roaming_gb,
            "exceso_datos_gb"  : exceso_datos_gb,
            "cargo_exceso_eur" : cargo_exceso,
            "fecha_mes"        : (datetime(2024, 3, 28) - timedelta(days=(12 - mes) * 30)).strftime("%Y-%m"),
        })

consumo = pd.DataFrame(consumo_rows)
consumo["consumo_id"] = [f"CNS{str(i).zfill(8)}" for i in range(1, len(consumo) + 1)]
print(f"   ✓ {len(consumo):,} registros de consumo generados")


# ═══════════════════════════════════════════════
# TABLA 4 — red_costes
# ═══════════════════════════════════════════════
print("📋 Generando tabla: red_costes...")

TIPOS_RED = ["4G", "5G", "4G/5G"]
PESOS_RED = [0.35, 0.25, 0.40]

red_costes = pd.DataFrame({
    "coste_id"              : [f"RED{str(i).zfill(6)}" for i in range(1, N_CLIENTES + 1)],
    "contrato_id"           : contratos["contrato_id"],
    "cliente_id"            : clientes["cliente_id"],
    "tipo_red_principal"    : np.random.choice(TIPOS_RED, N_CLIENTES, p=PESOS_RED),
    "cobertura_pct"         : np.random.normal(92, 5, N_CLIENTES).clip(60, 100).round(1),
    "coste_red_mensual_eur" : np.random.normal(6.5, 2.0, N_CLIENTES).clip(2, 15).round(2),
    "coste_interconexion_eur": np.random.normal(1.2, 0.4, N_CLIENTES).clip(0.2, 4).round(2),
    "coste_roaming_eur"     : np.random.normal(0.8, 0.5, N_CLIENTES).clip(0, 5).round(2),
    "margen_bruto_pct"      : np.random.normal(42, 8, N_CLIENTES).clip(15, 70).round(1),
    "congestion_red_pct"    : np.random.beta(2, 5, N_CLIENTES).round(3),
    "latencia_media_ms"     : np.random.normal(25, 8, N_CLIENTES).clip(5, 80).round(1),
})

print(f"   ✓ {len(red_costes):,} registros de costes de red generados")


# ═══════════════════════════════════════════════
# TABLA 5 — competencia_mercado
# ═══════════════════════════════════════════════
print("📋 Generando tabla: competencia_mercado...")

COMPETIDORES = ["Movistar", "Vodafone", "Orange", "MásMóvil", "Digi"]

competencia = pd.DataFrame({
    "mercado_id"               : [f"MKT{str(i).zfill(6)}" for i in range(1, N_CLIENTES + 1)],
    "cliente_id"               : clientes["cliente_id"],
    "segmento"                 : clientes["segmento"],
    "comunidad_autonoma"       : clientes["comunidad_autonoma"],
    "precio_competidor_min_eur": np.random.normal(12, 4, N_CLIENTES).clip(5, 30).round(2),
    "precio_competidor_max_eur": np.random.normal(45, 15, N_CLIENTES).clip(20, 120).round(2),
    "precio_medio_mercado_eur" : np.random.normal(28, 8, N_CLIENTES).clip(10, 80).round(2),
    "competidor_principal"     : np.random.choice(COMPETIDORES, N_CLIENTES),
    "indice_penetracion_mercado": np.random.beta(3, 2, N_CLIENTES).round(3),
    "elasticidad_precio_segmento": np.where(
        clientes["segmento"] == "Residencial", np.random.normal(-1.8, 0.3, N_CLIENTES).clip(-3, -0.5),
        np.where(clientes["segmento"] == "PYME", np.random.normal(-1.2, 0.2, N_CLIENTES).clip(-2, -0.3),
                 np.random.normal(-0.8, 0.2, N_CLIENTES).clip(-1.5, -0.2))
    ).round(3),
    "cuota_mercado_operador_pct": np.random.normal(28, 5, N_CLIENTES).clip(10, 50).round(1),
})

print(f"   ✓ {len(competencia):,} registros de mercado generados")


# ═══════════════════════════════════════════════
# GUARDAR — SQLite + CSVs
# ═══════════════════════════════════════════════
print("\n💾 Guardando datos...")

db_path = os.path.join(OUTPUT_DIR, "teleco_pricing.db")
conn = sqlite3.connect(db_path)

clientes.to_sql("clientes", conn, if_exists="replace", index=False)
contratos.to_sql("contratos", conn, if_exists="replace", index=False)
consumo.to_sql("consumo_mensual", conn, if_exists="replace", index=False)
red_costes.to_sql("red_costes", conn, if_exists="replace", index=False)
competencia.to_sql("competencia_mercado", conn, if_exists="replace", index=False)

conn.close()
print(f"   ✓ SQLite: {db_path}")

for nombre, df in [
    ("clientes", clientes), ("contratos", contratos),
    ("consumo_mensual", consumo), ("red_costes", red_costes),
    ("competencia_mercado", competencia)
]:
    path = os.path.join(OUTPUT_DIR, f"{nombre}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✓ CSV: {nombre}.csv  ({len(df):,} filas)")

# ─────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("✅  DATASET TELECOMUNICACIONES GENERADO CORRECTAMENTE")
print("═" * 60)
print(f"📁 Carpeta         : ./{OUTPUT_DIR}/")
print(f"🗄️  Base de datos   : teleco_pricing.db")
print(f"\n📊 Tablas:")
print(f"   • clientes              — {len(clientes):>8,} filas")
print(f"   • contratos             — {len(contratos):>8,} filas")
print(f"   • consumo_mensual       — {len(consumo):>8,} filas")
print(f"   • red_costes            — {len(red_costes):>8,} filas")
print(f"   • competencia_mercado   — {len(competencia):>8,} filas")
print(f"\n🎯 Variable objetivo: tarifa_contratada (€/mes)")
print(f"   Media  : {contratos['tarifa_contratada'].mean():.2f}€")
print(f"   Mínima : {contratos['tarifa_contratada'].min():.2f}€")
print(f"   Máxima : {contratos['tarifa_contratada'].max():.2f}€")
print(f"\n📱 Distribución por segmento:")
for seg in ["Residencial", "PYME", "Corporativo"]:
    mask = clientes["segmento"] == seg
    tarifa_seg = contratos[mask]["tarifa_contratada"]
    print(f"   {seg:<12}: {mask.sum():>6,} clientes | tarifa media: {tarifa_seg.mean():.2f}€")
print(f"\n🚀 Siguiente paso: abrir teleco_pricing.ipynb")
print("═" * 60)
