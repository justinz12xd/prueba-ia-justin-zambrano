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

# Los mensajes se componen de: apertura + núcleo + cierre.
#
# Las aperturas y los cierres son COMPARTIDOS por las tres clases a propósito. En la
# versión anterior cada clase tenía 8 frases fijas, y con tan poca variedad el modelo
# aprendió atajos: la palabra "una" aparecía solo en frases negativas ("esperando una
# respuesta"), así que cualquier mensaje con "una" se clasificaba como negativo con
# confianza 1.0 — incluido "Una última cosa, ¿cuál es el horario de atención?".
# Repartiendo las palabras funcionales entre las tres clases, la señal tiene que venir
# del contenido y no del relleno.
APERTURAS = [
    "", "", "", "Hola, ", "Buenas, ", "Buenos días, ", "Buenas tardes, ",
    "Una consulta, ", "Una última cosa, ", "Disculpe, ", "Por favor, ",
    "Hola de nuevo, ", "Buenas noches, ", "Estimados, ", "Una cosa más, ",
]
CIERRES = [
    "", "", "", ", gracias", ", muchas gracias", ", por favor", ", quedo atento",
    ", saludos", ", espero su respuesta", ", agradezco su ayuda", ". Gracias de antemano",
]

POSITIVE_CORES = [
    "quedó todo resuelto perfectamente",
    "excelente atención, el problema se solucionó muy rápido",
    "ya funciona el internet y estoy muy contento con el resultado",
    "muy buen servicio, el agente fue amable y resolvió todo",
    "eso era justo lo que necesitaba y quedó impecable",
    "quedé muy satisfecho con la solución que me dieron",
    "todo excelente, seguiré siendo cliente sin duda",
    "me encantó la rapidez con la que atendieron mi caso",
    "el técnico llegó puntual y dejó la instalación perfecta",
    "felicito al equipo por la atención recibida",
    "la velocidad del plan {plan} mejoró muchísimo",
    "resolvieron el cobro en menos de {dias} días, muy eficientes",
    "estoy encantado con el cambio a {mb} megas",
    "la persona que me atendió explicó todo con mucha claridad",
    "gracias por la gestión, superó mis expectativas",
    "el servicio en {ciudad} funciona mejor que nunca",
    "valió la pena esperar, quedó todo funcionando bien",
    "muy conforme con el descuento de {monto} dólares aplicado",
    "recomendaré su servicio a mis conocidos",
    "el acompañamiento durante todo el proceso fue excelente",
    "quedó resuelto al primer intento, sin complicaciones",
    "la atención por chat fue rápida y efectiva",
    "agradezco la solución, superaron lo que esperaba",
    "todo perfecto con la instalación del nuevo router",
]
NEUTRAL_CORES = [
    "quisiera confirmar el estado de mi solicitud",
    "necesito saber el horario de atención de las oficinas",
    "¿cuál es el procedimiento para actualizar mis datos?",
    "¿cuánto tiempo tarda en aplicarse este cambio?",
    "quedo pendiente de la respuesta del área técnica",
    "voy a revisar la información que me enviaron",
    "esperaré la llamada del técnico en el horario indicado",
    "¿me pueden indicar en qué fecha se emite la factura?",
    "necesito más información sobre el proceso de instalación",
    "¿el plan {plan} incluye instalación sin costo?",
    "quisiera saber si atienden los sábados en {ciudad}",
    "¿dónde puedo descargar mi comprobante de pago?",
    "consulto por los requisitos para contratar {mb} megas",
    "¿cuál es el número de contacto del área de soporte?",
    "quiero confirmar la dirección registrada en mi cuenta",
    "¿en cuántos días hábiles se procesa la solicitud?",
    "necesito el detalle de los cargos de este mes",
    "¿puedo cambiar la fecha de pago a fin de mes?",
    "me gustaría conocer las opciones disponibles",
    "¿qué documentos debo presentar para el trámite?",
    "solicito información sobre la cobertura en mi zona",
    "¿el cambio se aplica de inmediato o el próximo ciclo?",
    "quisiera agendar la visita técnica para la próxima semana",
    "de acuerdo, entendido, procedo como me indicaron",
    "¿cuál es el costo mensual del plan de {mb} megas?",
    "necesito verificar si mi pago de {monto} dólares fue registrado",
    "consulto el estado del ticket que abrí hace {dias} días",
    "¿hay alguna oficina de atención cerca de {ciudad}?",
    # Reportes de avería en tono CALMADO. Comparten vocabulario con la clase negativa
    # ("se corta", "no funciona", "sin señal") pero sin carga emocional: sin ellos el
    # modelo aprende que hablar de una avería es estar enfadado, y el agente termina
    # escalando a un humano cualquier reporte técnico normal.
    "el internet se corta cada media hora desde hace {dias} días",
    "reporto que el router no da señal desde ayer por la noche",
    "les comento que la conexión viene intermitente esta semana",
    "informo que no tengo servicio desde esta mañana",
    "la velocidad está por debajo de los {mb} megas contratados",
    "adjunto el detalle del cobro de {monto} dólares que quiero revisar",
    "el módem enciende la luz roja y no navega",
    "aviso que se cortó el servicio en {ciudad} desde el mediodía",
    "ya reinicié el router y el problema continúa",
    "les escribo para reportar una falla en la línea",
    "el servicio funciona lento en las horas de la tarde",
    "quisiera reportar que la señal se pierde por momentos",
]
NEGATIVE_CORES = [
    "esto es inaceptable, llevo {dias} días sin internet y nadie resuelve nada",
    "estoy muy molesto, ya es la tercera vez que reporto lo mismo",
    "pésimo servicio, quiero hablar con un supervisor urgente",
    "no puede ser que sigan cobrando mal después de tantos reclamos",
    "estoy realmente frustrado, nadie me da respuestas concretas",
    "es indignante el tiempo que llevo esperando",
    "ya no aguanto más este servicio, quiero cancelar ya mismo",
    "es la peor experiencia que he tenido con un proveedor",
    "me cobraron {monto} dólares de más y nadie me responde",
    "llevo {dias} días esperando al técnico que nunca llega",
    "estoy harto de repetir el mismo problema cada semana",
    "el servicio en {ciudad} es lamentable, se cae todo el tiempo",
    "me prometieron {mb} megas y no llegan ni a la mitad",
    "esto es una falta de respeto hacia el cliente",
    "estoy furioso, me cortaron el servicio sin aviso",
    "qué vergüenza de atención, nadie se hace responsable",
    "es inadmisible que el plan {plan} funcione tan mal",
    "estoy cansado de que me pasen de un área a otra",
    "un desastre, perdí toda la mañana esperando",
    "me siento estafado con lo que estoy pagando",
    "esto ya es el colmo, exijo una compensación",
    "muy decepcionado, esperaba mucho más de ustedes",
    "es intolerable la cantidad de cortes que tengo",
    "estoy indignado, cancelen mi contrato de inmediato",
]

SENTIMENT_CORES = {
    "positive": POSITIVE_CORES,
    "neutral": NEUTRAL_CORES,
    "negative": NEGATIVE_CORES,
}

AGENT_RESPONSES = [
    "Entiendo su situación, permítame revisar su cuenta para ayudarle.",
    "Gracias por contactarnos, en un momento verifico la información.",
    "Lamento el inconveniente, procedo a escalar su caso de inmediato.",
    "Con gusto le ayudo, ¿me confirma su número de cliente?",
    "Ya generé la solicitud, un técnico se comunicará en las próximas horas.",
    "Su pago fue verificado correctamente, no hay ninguna irregularidad.",
    "Registré su consulta, le responderemos por este mismo medio.",
    "Le comparto la información solicitada en el correo registrado.",
]

# Generador propio: así los mensajes de sentimiento pueden cambiar sin alterar los
# demás datasets, que dependen del flujo del generador global.
rng_msgs = np.random.default_rng(SEED + 1)


def _componer_mensaje(sentiment: str) -> str:
    """Apertura + núcleo + cierre, con las variables del núcleo rellenadas."""
    cores = SENTIMENT_CORES[sentiment]
    nucleo = cores[rng_msgs.integers(0, len(cores))]
    nucleo = nucleo.format(
        dias=rng_msgs.integers(2, 15),
        mb=rng_msgs.choice([50, 100, 200, 300, 500, 1000]),
        monto=rng_msgs.choice([5, 8, 12, 15, 20, 25, 35, 40]),
        plan=rng_msgs.choice(PLANES),
        ciudad=rng_msgs.choice(CIUDADES),
    )
    apertura = APERTURAS[rng_msgs.integers(0, len(APERTURAS))]
    cierre = CIERRES[rng_msgs.integers(0, len(CIERRES))]
    if not apertura:
        nucleo = nucleo[0].upper() + nucleo[1:]
    return f"{apertura}{nucleo}{cierre}"


def build_interactions(n: int = 2400) -> pd.DataFrame:
    clases = list(SENTIMENT_CORES)
    rows = []
    for i in range(1, n + 1):
        sentiment = clases[i % len(clases)]  # clases balanceadas
        msg = _add_noise_msgs(_componer_mensaje(sentiment))
        rows.append({
            "interaction_id": i,
            "ticket_id": int(rng_msgs.integers(1, 800)),
            "agent_response": AGENT_RESPONSES[rng_msgs.integers(0, len(AGENT_RESPONSES))],
            "customer_msg": msg,
            "sentiment": sentiment,
            "resolution_time": round(float(rng_msgs.exponential(6) + 0.5), 2),
        })
    return pd.DataFrame(rows)


def _add_noise_msgs(text: str) -> str:
    """Igual que _add_noise pero con el generador propio de los mensajes."""
    if rng_msgs.random() < 0.15:
        text = (text.replace("á", "a").replace("é", "e").replace("í", "i")
                     .replace("ó", "o").replace("ú", "u"))
    if rng_msgs.random() < 0.10:
        text = text[0].lower() + text[1:]
    if rng_msgs.random() < 0.10:
        text = text + "!!"
    return text


# ---------------------------------------------------------------------------
# 4) tickets_resolution.csv — tiempo de resolución (Parte 2.2)
# ---------------------------------------------------------------------------

BASE_HOURS = {"TECH": 8.0, "BILL": 3.0, "PLAN": 4.0, "CNCL": 5.0, "OTHR": 2.0}
PRIORITY_MULT = {"low": 1.4, "medium": 1.0, "high": 0.6}


# Generador propio, para que este dataset no dependa de cuántos números hayan
# consumido los anteriores: así cambiar los mensajes de sentimiento no altera los
# tiempos de resolución.
rng_res = np.random.default_rng(SEED + 2)


def build_resolution(tickets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in tickets_df.iterrows():
        hour = int(rng_res.integers(0, 24))
        weekday = int(rng_res.integers(0, 7))
        base = BASE_HOURS[t["category"]] * PRIORITY_MULT[t["priority"]]
        weekend_penalty = 1.3 if weekday >= 5 else 1.0
        night_penalty = 1.2 if hour >= 20 or hour <= 6 else 1.0
        resolution_hours = max(0.25, base * weekend_penalty * night_penalty *
                                float(rng_res.lognormal(mean=0, sigma=0.35)))
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
