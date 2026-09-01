"""Prueba manual del clasificador de tickets (Parte 1.1).

Carga el modelo entrenado directamente con joblib —sin levantar la API ni pedir
token— para poder probarlo en vivo con frases escritas a mano.

Uso:
    python scripts/probar_clasificador.py                      # batería de frases nuevas
    python scripts/probar_clasificador.py "mi wifi no anda"    # una o varias frases sueltas
    python scripts/probar_clasificador.py --interactivo        # escribir y clasificar en vivo
    python scripts/probar_clasificador.py --dataset            # evaluar data/tickets_train.csv

Con el stack de Docker levantado, sin instalar nada en la máquina:
    docker exec -it telecom_support_api python scripts/probar_clasificador.py --interactivo

NOTA sobre --dataset: el CSV incluye las filas con las que se entrenó el modelo,
así que su accuracy sale inflada. Las métricas honestas (CV 5-fold y test apartado)
están en saved_models/ml/ticket_classifier_report.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml_runtime import ticket_classifier  # noqa: E402

CATEGORIES = ["TECH", "BILL", "PLAN", "CNCL", "OTHR"]

NOMBRES = {
    "TECH": "Problema técnico",
    "BILL": "Facturación",
    "PLAN": "Cambio de plan",
    "CNCL": "Cancelación",
    "OTHR": "Otros",
}

# Frases escritas a mano, deliberadamente distintas de las plantillas del generador
# de datos sintéticos: es lo único que mide si el modelo generaliza de verdad.
CASOS_DE_PRUEBA: list[tuple[str, str]] = [
    ("TECH", "el wifi anda lentísimo en el cuarto del fondo"),
    ("TECH", "se me cayó la señal ayer en la noche y no volvió"),
    ("TECH", "el módem prende una luz roja y no navega nada"),
    ("BILL", "por qué este mes me llegó más caro que el anterior"),
    ("BILL", "necesito la factura del mes pasado en pdf"),
    ("BILL", "me hicieron un cobro que no reconozco"),
    ("PLAN", "quiero subir de 100 a 300 megas"),
    ("PLAN", "qué promociones tienen para agregar televisión"),
    ("CNCL", "ya no quiero el servicio, den de baja mi cuenta"),
    ("CNCL", "me mudo de país el mes que viene y necesito terminar el contrato"),
    ("OTHR", "solo quería felicitar al técnico que vino ayer"),
    ("OTHR", "cuál es el horario de atención de las oficinas"),
]


def barra(valor: float, ancho: int = 20) -> str:
    llenos = int(round(valor * ancho))
    return "█" * llenos + "·" * (ancho - llenos)


def clasificar(texto: str, detalle: bool = False) -> tuple[str, float]:
    categoria, probabilidades = ticket_classifier.classify_ticket(texto)
    confianza = probabilidades[categoria]
    if detalle:
        for cat, p in sorted(probabilidades.items(), key=lambda kv: kv[1], reverse=True):
            marca = "->" if cat == categoria else "  "
            print(f"    {marca} {cat:<5} {barra(p)} {p * 100:5.1f}%  {NOMBRES[cat]}")
    return categoria, confianza


def modo_bateria() -> int:
    """Corre los casos escritos a mano y devuelve el número de fallos."""
    print(f"\nClasificando {len(CASOS_DE_PRUEBA)} frases nuevas "
          "(no salen de las plantillas del generador)\n")
    print(f'{"esperado":<9} {"predicho":<9} {"conf":>6}   frase')
    print("-" * 88)

    fallos = []
    for esperado, frase in CASOS_DE_PRUEBA:
        predicho, confianza = clasificar(frase)
        acierta = predicho == esperado
        if not acierta:
            fallos.append((esperado, predicho, confianza, frase))
        estado = "ok   " if acierta else "FALLA"
        print(f"{esperado:<9} {predicho:<9} {confianza * 100:>5.1f}%  {estado} {frase}")

    aciertos = len(CASOS_DE_PRUEBA) - len(fallos)
    print("-" * 88)
    print(f"Aciertos: {aciertos}/{len(CASOS_DE_PRUEBA)} "
          f"({aciertos / len(CASOS_DE_PRUEBA) * 100:.0f}%)")

    if fallos:
        print("\nFallos, con el desglose de probabilidades:")
        for esperado, predicho, _, frase in fallos:
            print(f'\n  "{frase}"')
            print(f"  esperado {esperado} ({NOMBRES[esperado]}), "
                  f"predicho {predicho} ({NOMBRES[predicho]})")
            clasificar(frase, detalle=True)
        print("\n  Estos fallos son esperables: TF-IDF pondera palabras sueltas, así que una\n"
              "  palabra fuerte de otra categoría (\"técnico\", \"factura\") puede arrastrar la\n"
              "  predicción aunque la intención de la frase sea distinta.")
    return len(fallos)


def modo_frases(frases: list[str]) -> None:
    for frase in frases:
        print(f'\n"{frase}"')
        try:
            categoria, confianza = clasificar(frase, detalle=True)
            print(f"  => {categoria} ({NOMBRES[categoria]}) con {confianza * 100:.1f}% de confianza")
        except ValueError as exc:
            print(f"  !! {exc}")


def modo_interactivo() -> None:
    print("\nEscribe una descripción de ticket y pulsa Enter. Ctrl+C o línea vacía para salir.")
    print("(el modelo exige al menos 10 caracteres)\n")
    while True:
        try:
            frase = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not frase:
            return
        try:
            categoria, confianza = clasificar(frase, detalle=True)
            print(f"  => {categoria} ({NOMBRES[categoria]}) con {confianza * 100:.1f}% "
                  f"de confianza\n")
        except ValueError as exc:
            print(f"  !! {exc}\n")


def modo_dataset() -> None:
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    ruta = ROOT / "data" / "tickets_train.csv"
    if not ruta.exists():
        print(f"No existe {ruta}. Ejecuta antes: python scripts/generate_synthetic_data.py")
        return

    df = pd.read_csv(ruta).dropna(subset=["description", "category"])
    pipeline = ticket_classifier.get_pipeline()
    predicho = pipeline.predict(df["description"])

    print(f"\nEvaluando {len(df)} tickets de {ruta.name}\n")
    print(classification_report(df["category"], predicho, labels=CATEGORIES, zero_division=0))

    print("Matriz de confusión (filas = real, columnas = predicho)")
    matriz = confusion_matrix(df["category"], predicho, labels=CATEGORIES)
    print("        " + "  ".join(f"{c:>5}" for c in CATEGORIES))
    for categoria, fila in zip(CATEGORIES, matriz):
        print(f"  {categoria:<5} " + "  ".join(f"{v:>5}" for v in fila))

    errores = df[df["category"] != predicho]
    print(f"\nFallos: {len(errores)} de {len(df)}")
    for _, fila in errores.head(10).iterrows():
        pred = pipeline.predict([fila["description"]])[0]
        print(f'  real={fila["category"]} pred={pred} | {fila["description"][:70]}')

    print("\nOJO: este CSV incluye las filas de entrenamiento, así que la accuracy sale\n"
          "inflada. Las métricas honestas (CV 5-fold y test apartado) están en\n"
          "saved_models/ml/ticket_classifier_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prueba el clasificador de tickets con frases propias.",
        epilog="Sin argumentos corre una batería de frases nuevas escritas a mano.",
    )
    parser.add_argument("frases", nargs="*", help="Frases a clasificar")
    parser.add_argument("-i", "--interactivo", action="store_true",
                        help="Modo interactivo: escribir frases y ver la predicción")
    parser.add_argument("-d", "--dataset", action="store_true",
                        help="Evaluar data/tickets_train.csv completo")
    args = parser.parse_args()

    try:
        ticket_classifier.get_pipeline()
    except ticket_classifier.TicketClassifierUnavailable as exc:
        print(f"No se pudo cargar el modelo: {exc}")
        raise SystemExit(1)

    if args.dataset:
        modo_dataset()
    elif args.interactivo:
        modo_interactivo()
    elif args.frases:
        modo_frases(args.frases)
    else:
        modo_bateria()


if __name__ == "__main__":
    main()
