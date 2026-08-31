"""
Generador de datos sintéticos para el Sistema Inteligente de Atención al Cliente.

Produce, dentro de data/:
  - tickets_train.csv     -> descripción de ticket + categoría (Parte 1.1)
  - customers.csv         -> features de clientes + churn_status (Parte 1.2)
  - interactions.csv      -> mensajes cliente-agente + sentimiento (Parte 2.1)
  - tickets_resolution.csv-> features de ticket + tiempo de resolución en horas (Parte 2.2)

Todo se genera con una semilla fija (seed=42) para reproducibilidad. El texto es
español informal, con tildes, mayúsculas mixtas y algo de ruido (typos), simulando
mensajes reales de clientes.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
rng = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PLANES = ["Fibra 200MB", "Fibra 500MB", "Internet Hogar 100MB", "Plan Familiar",
          "Plan Ilimitado", "Combo TV+Internet", "Plan Prepago Movil", "Fibra 1GB"]
CIUDADES = ["Quito", "Guayaquil", "Cuenca", "Ambato", "Manta", "Loja", "Riobamba"]

# ---------------------------------------------------------------------------
# 1) tickets_train.csv — clasificación de tickets
# ---------------------------------------------------------------------------

TECH_TEMPLATES = [
    "El internet está muy lento desde hace {dias} días, no puedo ni hacer una videollamada",
    "No tengo conexión a internet, el router tiene la luz roja parpadeando",
    "Se me corta el wifi cada {min} minutos, es muy frustrante",
    "La velocidad contratada es de {mb}MB pero el speedtest me marca menos de la mitad",
    "Mi router no enciende, ya lo reinicié varias veces y sigue sin funcionar",
    "Tengo pérdida de paquetes constante jugando online, revisen la señal por favor",
    "No carga ninguna página web desde esta mañana en {ciudad}",
    "El servicio de internet se cae todas las noches entre las 8 y las 10",
    "La luz del módem está en ámbar, no logro navegar",
    "Contraté fibra óptica pero la velocidad real es pésima, parece adsl",
    "No puedo configurar el wifi de 5GHz en mi router nuevo",
    "Se me desconecta el internet apenas empieza a llover, necesito soporte técnico",
    "El técnico vino pero el problema de conexión sigue igual",
    "Tengo alta latencia y ping inestable trabajando desde casa",
    "No detecta la red wifi en ninguno de mis dispositivos desde ayer",
]

BILL_TEMPLATES = [
    "Me llegó un cobro duplicado en la factura de este mes, favor revisar",
    "No reconozco un cargo de ${monto} en mi última factura",
    "Quisiera una copia de mi factura del mes pasado para mi contabilidad",
    "El descuento de la promoción no se aplicó en mi factura, sigo pagando el full",
    "Mi factura llegó con un monto mayor al acordado en el contrato",
    "Necesito cambiar mi método de pago de efectivo a débito automático",
    "¿Por qué me cobraron ${monto} de más este mes en {ciudad}?",
    "Quiero saber la fecha límite de pago de mi factura actual",
    "Se me generó un cargo por mora que no debería aplicar, ya pagué a tiempo",
    "El valor de mi plan subió sin previo aviso, quiero una explicación",
    "No me ha llegado la factura electrónica de este mes a mi correo",
    "Solicito el desglose detallado de los cargos de mi factura",
    "Me cobraron un servicio adicional que nunca contraté",
    "Quiero verificar si mi pago de ayer ya se refleja en el sistema",
    "El total de la factura no coincide con el valor de mi plan mensual",
]

PLAN_TEMPLATES = [
    "Quiero cambiar mi plan actual a {plan}, ¿cuáles son los requisitos?",
    "Me interesa subir de velocidad, ¿qué opciones tienen sobre {mb}MB?",
    "Quisiera agregar el servicio de TV por cable a mi plan de internet",
    "¿Puedo hacer upgrade a {plan} sin pagar penalidad por cambio?",
    "Necesito bajar de plan porque el actual me queda muy caro",
    "Quiero agregar una línea móvil adicional a mi plan familiar",
    "¿Tienen promociones para migrar a {plan} este mes?",
    "Deseo conocer los beneficios de pasar de plan mes a mes a plan anual",
    "Quiero contratar el {plan} para mi nueva casa en {ciudad}",
    "¿Cómo hago para cambiar de plan prepago a pospago?",
    "Estoy interesado en el combo de internet más streaming",
    "Quiero información sobre los planes empresariales disponibles",
    "¿Puedo combinar dos servicios en un solo plan con descuento?",
    "Necesito aumentar la cantidad de datos móviles de mi plan actual",
    "Quisiera cambiar mi plan antes de que termine el contrato actual",
]

CNCL_TEMPLATES = [
    "Quiero cancelar mi servicio de internet, ya no lo necesito",
    "Deseo dar de baja mi línea porque me mudo de ciudad",
    "Estoy muy insatisfecho con el servicio y quiero cancelar el contrato",
    "¿Cuál es el proceso para cancelar mi plan sin penalidad?",
    "Ya encontré otro proveedor, necesito cancelar mi cuenta cuanto antes",
    "No estoy conforme con la atención recibida, quiero terminar el contrato",
    "Quiero dar de baja el combo de TV pero mantener el internet",
    "Llevo meses con problemas y ya decidí cancelar el servicio",
    "¿Tiene costo cancelar antes de que termine el contrato de {plan}?",
    "Necesito cancelar el servicio a nombre de un familiar fallecido",
    "Quiero cancelar porque el precio subió demasiado",
    "Solicito la baja definitiva de todos mis servicios contratados",
    "Me mudo fuera de {ciudad} y no hay cobertura, debo cancelar",
    "Quiero cancelar mi suscripción antes de la próxima facturación",
    "Ya no puedo pagar el servicio, necesito darlo de baja este mes",
]

OTHR_TEMPLATES = [
    "¿Cuál es el horario de atención de las oficinas en {ciudad}?",
    "Quisiera saber si tienen cobertura en el sector de {ciudad}",
    "¿Dónde queda la sucursal más cercana a mi domicilio?",
    "Tengo una consulta general sobre los servicios que ofrecen",
    "¿Puedo pagar mi factura en efectivo en algún punto físico?",
    "Quiero felicitar al agente que me atendió, excelente servicio",
    "¿Qué documentos necesito para abrir una cuenta nueva?",
    "Necesito hablar con un asesor sobre un tema que no es técnico ni de facturación",
    "¿Tienen aplicación móvil para gestionar mi cuenta?",
    "Quisiera saber más sobre las políticas de privacidad de datos",
    "¿Cómo puedo actualizar mis datos de contacto?",
    "Tengo una duda sobre el proceso de portabilidad numérica",
    "¿Ofrecen algún programa de referidos con beneficios?",
    "Quiero saber si puedo transferir mi cuenta a otra persona",
    "Consulta general, no encuentro la opción que busco en la app",
]

CATEGORY_TEMPLATES = {
    "TECH": TECH_TEMPLATES,
    "BILL": BILL_TEMPLATES,
    "PLAN": PLAN_TEMPLATES,
    "CNCL": CNCL_TEMPLATES,
    "OTHR": OTHR_TEMPLATES,
}


def _fill(template: str) -> str:
    return template.format(
        dias=rng.integers(1, 10),
        min=rng.choice([5, 10, 15, 20, 30]),
        mb=rng.choice([50, 100, 200, 300, 500, 1000]),
        monto=rng.choice([5, 8, 12, 15, 20, 25, 35]),
        plan=rng.choice(PLANES),
        ciudad=rng.choice(CIUDADES),
    )


def _add_noise(text: str) -> str:
    """Simula errores humanos leves: quitar tildes o cambiar mayúsculas ocasionalmente."""
    if rng.random() < 0.15:
        text = (text.replace("á", "a").replace("é", "e").replace("í", "i")
                     .replace("ó", "o").replace("ú", "u"))
    if rng.random() < 0.1:
        text = text[0].lower() + text[1:]
    if rng.random() < 0.1:
        text = text + "!!"
    return text


def build_tickets(n_per_category: int = 160) -> pd.DataFrame:
    rows = []
    tid = 1
    for category, templates in CATEGORY_TEMPLATES.items():
        for _ in range(n_per_category):
            template = rng.choice(templates)
            desc = _add_noise(_fill(template))
            priority = rng.choice(["low", "medium", "high"], p=[0.4, 0.4, 0.2])
            rows.append({"ticket_id": tid, "description": desc, "category": category,
                         "priority": priority})
            tid += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["ticket_id"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# 2) customers.csv — churn
# ---------------------------------------------------------------------------

def build_customers(n: int = 2000) -> pd.DataFrame:
    tenure_months = rng.integers(0, 73, size=n)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], size=n,
                                p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["credit_card", "bank_transfer", "cash"], size=n,
                                 p=[0.45, 0.35, 0.20])
    monthly_charge = np.round(rng.normal(45, 18, size=n).clip(15, 150), 2)
    total_charges = np.round(monthly_charge * tenure_months * rng.uniform(0.9, 1.05, size=n), 2)
    num_tickets = rng.poisson(1.2, size=n).clip(0, 15)
    avg_satisfaction = np.round(rng.normal(3.6, 1.1, size=n).clip(1, 5), 2)
    plan_type = rng.choice(PLANES, size=n)

    # Introduce nulls (missing data) a propósito para forzar manejo de nulos
    null_mask = rng.random(n) < 0.03
    avg_satisfaction = avg_satisfaction.astype(object)
    avg_satisfaction[null_mask] = None

    # Probabilidad de churn dependiente de features (relación realista + ruido)
    contract_risk = np.select(
        [contract_type == "month-to-month", contract_type == "one-year",
         contract_type == "two-year"],
        [0.35, 0.05, -0.05],
    )
    satisfaction_num = pd.to_numeric(pd.Series(avg_satisfaction), errors="coerce").fillna(3.6).values
    logit = (
        -1.6
        + contract_risk * 3.0
        - 0.02 * tenure_months
        + 0.012 * monthly_charge
        + 0.28 * num_tickets
        - 0.45 * satisfaction_num
        + rng.normal(0, 0.6, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn_status = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "name": [f"Cliente {i}" for i in range(1, n + 1)],
        "email": [f"cliente{i}@example.com" for i in range(1, n + 1)],
        "phone": ["09" + "".join(rng.choice(list("0123456789"), 8)) for _ in range(n)],
        "plan_type": plan_type,
        "monthly_charge": monthly_charge,
        "tenure_months": tenure_months,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "num_tickets": num_tickets,
        "avg_satisfaction": avg_satisfaction,
        "churn_status": churn_status,
    })
    return df


# ---------------------------------------------------------------------------
# 3) interactions.csv — sentimiento (Parte 2.1)
# ---------------------------------------------------------------------------

POSITIVE_MSGS = [
    "Muchas gracias, quedó todo resuelto perfectamente",
    "Excelente atención, el problema se solucionó muy rápido",
    "Genial, ya funciona el internet, gracias por la ayuda",
    "Muy buen servicio, el agente fue muy amable y resolvió todo",
    "Perfecto, eso era justo lo que necesitaba, gracias",
    "Quedé muy satisfecho con la solución que me dieron",
    "Todo excelente, seguiré siendo cliente sin duda",
    "Me encantó la rapidez con la que atendieron mi caso",
]
NEUTRAL_MSGS = [
    "Ok, entendido, voy a revisar la información enviada",
    "¿Podrían confirmarme el horario de atención mañana?",
    "Necesito más información sobre el proceso, por favor",
    "Está bien, quedo pendiente de la respuesta",
    "¿Cuánto tiempo tarda en aplicarse este cambio?",
    "Solo quería confirmar el estado de mi solicitud",
    "De acuerdo, esperaré la llamada del técnico",
    "Entiendo, gracias por la información proporcionada",
]
NEGATIVE_MSGS = [
    "Esto es inaceptable, llevo tres días sin internet y nadie resuelve nada",
    "Estoy muy molesto, ya es la tercera vez que reporto lo mismo",
    "Pésimo servicio, quiero hablar con un supervisor urgente",
    "No puede ser que sigan cobrando mal después de tantos reclamos",
    "Estoy realmente frustrado, nadie me da una solución definitiva",
    "Es indignante el tiempo que llevo esperando una respuesta",
    "Ya no aguanto más este servicio, quiero cancelar ya mismo",
    "Es la peor experiencia que he tenido con un proveedor de internet",
]
AGENT_RESPONSES = [
    "Entiendo su situación, permítame revisar su cuenta para ayudarle.",
    "Gracias por contactarnos, en un momento verifico la información.",
    "Lamento el inconveniente, procedo a escalar su caso de inmediato.",
    "Con gusto le ayudo, ¿me confirma su número de cliente?",
    "Ya generé la solicitud, un técnico se comunicará en las próximas horas.",
    "Su pago fue verificado correctamente, no hay ninguna irregularidad.",
]

SENTIMENT_POOL = (
    [(m, "positive") for m in POSITIVE_MSGS] +
    [(m, "neutral") for m in NEUTRAL_MSGS] +
    [(m, "negative") for m in NEGATIVE_MSGS]
)


def build_interactions(n: int = 1800) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        msg, sentiment = SENTIMENT_POOL[rng.integers(0, len(SENTIMENT_POOL))]
        msg = _add_noise(msg)
        rows.append({
            "interaction_id": i,
            "ticket_id": int(rng.integers(1, 800)),
            "agent_response": rng.choice(AGENT_RESPONSES),
            "customer_msg": msg,
            "sentiment": sentiment,
            "resolution_time": round(float(rng.exponential(6) + 0.5), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) tickets_resolution.csv — tiempo de resolución (Parte 2.2)
# ---------------------------------------------------------------------------

BASE_HOURS = {"TECH": 8.0, "BILL": 3.0, "PLAN": 4.0, "CNCL": 5.0, "OTHR": 2.0}
PRIORITY_MULT = {"low": 1.4, "medium": 1.0, "high": 0.6}


def build_resolution(tickets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in tickets_df.iterrows():
        hour = int(rng.integers(0, 24))
        weekday = int(rng.integers(0, 7))
        base = BASE_HOURS[t["category"]] * PRIORITY_MULT[t["priority"]]
        weekend_penalty = 1.3 if weekday >= 5 else 1.0
        night_penalty = 1.2 if hour >= 20 or hour <= 6 else 1.0
        resolution_hours = max(0.25, base * weekend_penalty * night_penalty *
                                float(rng.lognormal(mean=0, sigma=0.35)))
        rows.append({
            "ticket_id": t["ticket_id"],
            "category": t["category"],
            "priority": t["priority"],
            "description": t["description"],
            "hour_of_day": hour,
            "day_of_week": weekday,
            "resolution_time_hours": round(resolution_hours, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    tickets = build_tickets()
    tickets.to_csv(DATA_DIR / "tickets_train.csv", index=False, encoding="utf-8")
    print(f"tickets_train.csv -> {len(tickets)} filas")

    customers = build_customers()
    customers.to_csv(DATA_DIR / "customers.csv", index=False, encoding="utf-8")
    print(f"customers.csv -> {len(customers)} filas")

    interactions = build_interactions()
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False, encoding="utf-8")
    print(f"interactions.csv -> {len(interactions)} filas")

    resolution = build_resolution(tickets)
    resolution.to_csv(DATA_DIR / "tickets_resolution.csv", index=False, encoding="utf-8")
    print(f"tickets_resolution.csv -> {len(resolution)} filas")


if __name__ == "__main__":
    main()
