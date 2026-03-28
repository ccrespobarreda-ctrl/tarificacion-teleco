"""
API REST — Tarificación Dinámica Telecomunicaciones
====================================================
Endpoint principal: POST /v1/pricing
Devuelve: tarifa recomendada, rango de negociación e intervalo de confianza

Ejecutar en local:
    uvicorn main:app --reload --port 8000

Documentación automática:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pickle
import numpy as np
import pandas as pd
import time
import os

# ─────────────────────────────────────────────
# Cargar modelo al arrancar la API
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_tarificacion_v1.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        artefactos = pickle.load(f)
    modelo          = artefactos["modelo"]
    target_encoder  = artefactos["target_encoder"]
    features        = artefactos["features"]
    metricas_modelo = artefactos["metricas"]
    mae_segmento    = artefactos["mae_segmento"]
    print(f"✅ Modelo cargado | MAE global: {metricas_modelo['mae']}€ | R²: {metricas_modelo['r2']}")
except FileNotFoundError:
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}.")


# ─────────────────────────────────────────────
# Schemas de entrada y salida
# ─────────────────────────────────────────────

class PerfilCliente(BaseModel):
    """Perfil del cliente y datos de la solicitud de contrato"""

    # Datos del cliente
    edad                  : int   = Field(..., ge=18, le=100)
    segmento              : Literal["Residencial", "PYME", "Corporativo"]
    comunidad_autonoma    : str   = Field(..., description="Comunidad autónoma")
    canal_adquisicion     : Literal["Online", "Tienda física", "Teléfono", "Distribuidor"]
    ingreso_estimado      : float = Field(..., gt=0, description="Ingresos estimados mensuales (€)")
    antiguedad_meses      : int   = Field(..., ge=0)
    num_lineas_activas    : int   = Field(..., ge=1, le=500)
    tiene_fibra           : int   = Field(..., ge=0, le=1)
    tiene_tv              : int   = Field(..., ge=0, le=1)
    nps_score             : int   = Field(..., ge=0, le=10)

    # Datos del contrato solicitado
    plan_contratado       : Literal[
        "Básico 5GB", "Estándar 20GB", "Premium 50GB", "Ilimitado",
        "PYME Básico 20GB", "PYME Pro 50GB", "PYME Total Ilimitado",
        "Corp Standard", "Corp Advanced", "Corp Enterprise"
    ]
    datos_contratados_gb  : int   = Field(..., ge=1)
    descuento_aplicado    : float = Field(0.0, ge=0, le=0.5, description="Descuento en tanto por uno (0.15 = 15%)")
    permanencia_meses     : int   = Field(0, ge=0, le=36)
    roaming_activo        : int   = Field(0, ge=0, le=1)
    seguro_dispositivo    : int   = Field(0, ge=0, le=1)

    # Consumo histórico (si existe)
    consumo_medio_gb      : float = Field(..., ge=0)
    consumo_max_gb        : float = Field(..., ge=0)
    minutos_medios        : float = Field(180.0, ge=0)
    roaming_medio_gb      : float = Field(0.0, ge=0)
    total_excesos_eur     : float = Field(0.0, ge=0)
    exceso_medio_gb       : float = Field(0.0, ge=0)

    # Red
    tipo_red_principal    : Literal["4G", "5G", "4G/5G"] = "4G/5G"
    cobertura_pct         : float = Field(95.0, ge=0, le=100)
    coste_red_mensual_eur : float = Field(6.5, ge=0)
    coste_interconexion_eur: float = Field(1.2, ge=0)
    margen_bruto_pct      : float = Field(42.0, ge=0, le=100)
    congestion_red_pct    : float = Field(0.2, ge=0, le=1)
    latencia_media_ms     : float = Field(25.0, ge=0)

    # Mercado
    precio_competidor_min_eur : float = Field(12.0, ge=0)
    precio_competidor_max_eur : float = Field(45.0, ge=0)
    precio_medio_mercado_eur  : float = Field(28.0, ge=0)
    elasticidad_precio_segmento: float = Field(-1.5, le=0)
    indice_penetracion_mercado : float = Field(0.6, ge=0, le=1)
    cuota_mercado_operador_pct : float = Field(28.0, ge=0, le=100)

    class Config:
        schema_extra = {
            "example": {
                "edad": 42, "segmento": "PYME",
                "comunidad_autonoma": "Madrid",
                "canal_adquisicion": "Online",
                "ingreso_estimado": 4500.0,
                "antiguedad_meses": 36,
                "num_lineas_activas": 8,
                "tiene_fibra": 1, "tiene_tv": 0, "nps_score": 7,
                "plan_contratado": "PYME Pro 50GB",
                "datos_contratados_gb": 50,
                "descuento_aplicado": 0.10,
                "permanencia_meses": 24,
                "roaming_activo": 1, "seguro_dispositivo": 0,
                "consumo_medio_gb": 38.5, "consumo_max_gb": 48.2,
                "minutos_medios": 220.0, "roaming_medio_gb": 1.2,
                "total_excesos_eur": 12.5, "exceso_medio_gb": 0.3,
                "tipo_red_principal": "4G/5G",
                "cobertura_pct": 96.0, "coste_red_mensual_eur": 7.2,
                "coste_interconexion_eur": 1.4, "margen_bruto_pct": 44.0,
                "congestion_red_pct": 0.18, "latencia_media_ms": 22.0,
                "precio_competidor_min_eur": 18.0,
                "precio_competidor_max_eur": 55.0,
                "precio_medio_mercado_eur": 32.0,
                "elasticidad_precio_segmento": -1.2,
                "indice_penetracion_mercado": 0.65,
                "cuota_mercado_operador_pct": 27.0
            }
        }


class RespuestaPricing(BaseModel):
    """Respuesta del modelo de tarificación"""
    tarifa_recomendada_eur    : float
    rango_negociacion         : dict
    intervalo_confianza       : dict
    segmento                  : str
    mae_segmento_eur          : float
    decision_descuento        : str
    latencia_ms               : float
    version_modelo            : str
    metricas_modelo           : dict


# ─────────────────────────────────────────────
# Inicializar FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Tarificación Dinámica Telecomunicaciones API",
    description="""
    API REST para recomendación automática de tarifas en contratos de datos móviles.

    ## Uso
    - **POST /v1/pricing** — Recomienda la tarifa óptima para un perfil de cliente
    - **GET /v1/health** — Estado de la API y métricas del modelo
    - **GET /v1/modelo/info** — Información técnica del modelo

    ## Modelo
    XGBoost optimizado con Optuna (150 trials) | MAE: 4.06€ | R²: 0.89
    """,
    version="1.0.0",
)


# ─────────────────────────────────────────────
# Función de preprocesamiento
# ─────────────────────────────────────────────
def preparar_features(perfil: PerfilCliente) -> pd.DataFrame:
    datos = perfil.dict()

    # Feature engineering (igual que en el notebook)
    datos["ratio_consumo_contratado"]   = round(datos["consumo_medio_gb"] / max(datos["datos_contratados_gb"], 1), 4)
    datos["coste_total_mensual"]        = round(datos["coste_red_mensual_eur"] + datos["coste_interconexion_eur"], 2)
    datos["indice_multiproducto"]       = datos["tiene_fibra"] + datos["tiene_tv"] + (1 if datos["num_lineas_activas"] > 1 else 0)
    datos["valor_cliente_estimado"]     = round(datos["ingreso_estimado"] * datos["antiguedad_meses"] / 1000, 2)
    datos["saturacion_datos"]           = round(min(datos["consumo_medio_gb"] / max(datos["datos_contratados_gb"], 1), 1.5), 4)
    datos["diferencial_vs_mercado"]     = 0.0  # no disponible en el momento de la oferta

    categoricas    = ["segmento", "comunidad_autonoma", "canal_adquisicion",
                      "plan_contratado", "tipo_red_principal"]
    numericas_mod  = [f for f in features if f not in categoricas]

    df = pd.DataFrame([datos])[numericas_mod + categoricas]
    df = target_encoder.transform(df)
    return df


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {"mensaje": "Tarificación Dinámica Telecomunicaciones API v1.0", "docs": "/docs"}


@app.get("/v1/health", tags=["Monitoring"])
def health_check():
    return {
        "status"        : "healthy",
        "modelo_cargado": True,
        "metricas"      : metricas_modelo,
        "mae_segmento"  : mae_segmento,
        "version"       : "1.0.0"
    }


@app.get("/v1/modelo/info", tags=["Monitoring"])
def modelo_info():
    return {
        "algoritmo"              : "XGBoost",
        "optimizacion"           : "Optuna 150 trials",
        "features_utilizadas"    : len(features),
        "metricas_evaluacion"    : metricas_modelo,
        "mae_por_segmento"       : mae_segmento,
        "encoding_categoricas"   : "Target Encoding",
        "validacion"             : "K-Fold CV k=5",
        "dataset_entrenamiento"  : "95.000 contratos | 5 tablas relacionadas"
    }


@app.post("/v1/pricing", response_model=RespuestaPricing, tags=["Pricing"])
def recomendar_tarifa(perfil: PerfilCliente):
    """
    Recomienda la tarifa óptima para un contrato de datos móviles.

    Devuelve la tarifa recomendada, rango de negociación aceptable
    e intervalo de confianza basado en el MAE del segmento.
    """
    try:
        inicio = time.time()

        X             = preparar_features(perfil)
        tarifa_pred   = float(modelo.predict(X)[0])
        tarifa_pred   = round(max(5.0, tarifa_pred), 2)

        # Rango de negociación basado en MAE del segmento
        mae_seg = mae_segmento.get(perfil.segmento, metricas_modelo["mae"])
        rango = {
            "minimo_aceptable" : round(max(5.0, tarifa_pred - mae_seg), 2),
            "recomendada"      : tarifa_pred,
            "maximo_sugerido"  : round(tarifa_pred + mae_seg, 2)
        }

        # Intervalo de confianza 95%
        margen_ic = mae_seg * 1.96
        intervalo = {
            "inferior_95": round(max(5.0, tarifa_pred - margen_ic), 2),
            "superior_95": round(tarifa_pred + margen_ic, 2)
        }

        # Recomendación sobre descuento
        if perfil.descuento_aplicado == 0:
            decision_descuento = "Sin descuento aplicado — tarifa de precio completo"
        elif perfil.descuento_aplicado <= 0.10:
            decision_descuento = f"Descuento del {perfil.descuento_aplicado:.0%} — dentro del rango estándar"
        elif perfil.descuento_aplicado <= 0.20:
            decision_descuento = f"Descuento del {perfil.descuento_aplicado:.0%} — requiere aprobación supervisor"
        else:
            decision_descuento = f"Descuento del {perfil.descuento_aplicado:.0%} — descuento excepcional, requiere dirección comercial"

        latencia = round((time.time() - inicio) * 1000, 2)

        return RespuestaPricing(
            tarifa_recomendada_eur = tarifa_pred,
            rango_negociacion      = rango,
            intervalo_confianza    = intervalo,
            segmento               = perfil.segmento,
            mae_segmento_eur       = round(mae_seg, 2),
            decision_descuento     = decision_descuento,
            latencia_ms            = latencia,
            version_modelo         = "1.0.0",
            metricas_modelo        = metricas_modelo
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
