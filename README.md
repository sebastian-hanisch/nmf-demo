# NMF – Nicht-Negativität statt Unabhängigkeit – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-nmf-demo.streamlit.app/)**

Siebtes und **letztes Stück der Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **nicht-negative Matrixfaktorisierung (NMF)** – an einem wachsenden Beispiel,
mit **ICA**, **SOBI** und **SCA** als Vergleich (dazu PCA, die beste Einzelelektrode und eine bloße Schwelle als Referenzen). Vehikel: dasselbe **Mehrelektroden-Array** wie in [ica-demo](../ica-demo), [sobi-demo](../sobi-demo), [sca-demo](../sca-demo),
[spike-sorting-demo](../spike-sorting-demo) und [template-matching-demo](../template-matching-demo) (dort übernommen, per Test gegen eingefrorene Werte geprüft); neu sind die **Synchronität** der Neuronen (abhängige Quellen) und Wellenform-Ähnlichkeit bis 2 (sehr verschieden breite Spitzen).

**Einordnung in die Reihe (die Kanten des Graphen):** die ICA-Wurzel verlangt, dass die Quellen **statistisch unabhängig** sind. Die NMF lockert das zu **Nicht-Negativität**: eine Matrix V ≥ 0 wird als Produkt V ≈ W·H zweier nicht-negativer Matrizen geschrieben –
nichts wird abgezogen, es entstehen "Teile". Sie verlangt keine Unabhängigkeit und ist über das **Spektrogramm** auch mit **einer** Elektrode möglich (ICA, SOBI und SCA brauchen mehrere). Elektrodensignale sind aber bipolar (negative Spitze, positiver Nachschlag) und müssen erst **gleichgerichtet** werden;
ob das trägt, und was die NMF dann kann, ist die Frage dieser Demo. Die Antwort ist gemischt: **im Standardfall trennt sie schlechter als ICA, SOBI und SCA**, sie gewinnt bei abhängigen Quellen (mit Einschränkung) und – über das Spektrogramm – mit einer Elektrode bei sehr verschieden breiten Spitzen.
```
ica-demo → sobi-demo → sca-demo
        ↘ nmf-demo (Nicht-Negativität statt Unabhängigkeit; einkanalfähig)
pca-demo + Clustering-Linie → spike-sorting-demo → template-matching-demo → delay-graph-demo
```

| Frage | Ergebnis (4 Neuronen, 4 Elektroden, Rauschen 0.05, 20000 Abtastwerte; Mittel über 5 feste Datensätze, Seeds 100000–100004; Schwelle 2 Rausch-σ, Zufallsstart mit 5 Neustarts) |
|---|---|
| Standardfall | ❌ Spitzen-F1 der NMF **0.77**, ICA / SOBI / SCA **1.00** (PCA 0.61, beste Einzelelektrode 0.65). Die NMF findet eine Zerlegung, die die gleichgerichteten Daten **besser** erklärt (Fehler 0.013) als die wahre (0.049) – aber die falsche |
| Neuronenzahl | ❌ NMF 0.87 / 0.84 / 0.77 / 0.59 bei 2 / 3 / 4 / 5 Neuronen (ICA 1.00 / 1.00 / 1.00 / 0.77, SCA 0.99–1.00) – die positiven Mischspalten liegen dicht beieinander (größte Kosinus-Ähnlichkeit zweier Spalten 0.90) |
| Elektroden | ⚠️ 2 Elektroden **0.64** (ICA 0.36, SOBI 0.35, SCA 0.97), 3 Elektroden 0.72, ab 4 Elektroden 0.77; **eine Elektrode 0.38** (Schwelle 0.40, ICA 0.16, SOBI 0.16, SCA 0.05) |
| Synchrones Feuern | ✅ bei 0 / 30 / 60 / 90 % gemeinsamen Spikes NMF **0.77 / 0.87 / 0.91 / 0.98**, ICA 1.00 / 1.00 / 0.94 / **0.66**, SOBI 1.00 / 1.00 / 0.99 / 0.80, SCA 1.00 / 1.00 / 0.92 / 0.72. **Aber:** bei 90 % erreicht auch die beste Einzelelektrode 0.92 – bei so viel Gleichzeitigkeit ist die Aufgabe leicht |
| Rauschen | ⚠️ NMF (Schwelle 2) 0.77 / 0.75 / 0.71 / 0.71 bei 0.05 / 0.2 / 0.5 / 1.0; ICA 1.00 / 0.99 / 0.83 / 0.43, SOBI 1.00 / 0.79 / 0.53 / 0.38, SCA 1.00 / 1.00 / 0.94 / 0.86 – die NMF verliert wenig, SCA bleibt besser |
| Rauschschwelle der Gleichrichtung | ✅ bei Rauschen 1.0: **0.52 / 0.63 / 0.71 / 0.74** bei Schwelle 0 / 1 / 2 / 3 σ – ohne Schwelle wird das Rauschen mitgleichgerichtet und bildet eine positive Grundlinie; bei Rauschen 0.05 kaum ein Unterschied (0.76 gegen 0.77) |
| Gleichrichtung | nur die negative Spitze 0.77, Betrag beider Phasen 0.78 (Spitzen-F1), aber die **Korrelation** mit den wahren Neuronen fällt von 0.79 auf 0.55: Spitze und Nachschlag werden zu einer Aktivität verschmolzen |
| Sparsität λ | ⚠️ 0 / 0.5 / 1 / 3: 0.77 / 0.81 / 0.80 / 0.79, Mischspalten-Kosinus 0.95 / 0.98 / 0.98 / 0.98 – hilft ein wenig, der Fehler steigt dabei auf 0.086 (über den 0.049 der Wahrheit) |
| Iterationen | ❌ 20 / 100 / 300 / 1000: Fehler 0.055 / 0.017 / 0.013 / 0.010, aber Spitzen-F1 **0.75 / 0.80 / 0.77 / 0.74** – mehr Anpassung, schlechtere Trennung |
| Initialisierung | Zufallsstarts im Mittel 0.76 (Streuung um 0.03), der mit dem kleinsten Fehler 0.77, **NNDSVD 0.73** (höherer Fehler) |
| Komponentenzahl unbekannt | ⚠️ der Knick der Fehlerkurve trifft bei 2, 3 und 4 Neuronen in **allen fünf** Datensätzen, bei 5 Neuronen wählt er **immer 4** (Spitzen-F1 0.51 statt 0.61 mit richtiger Zahl; ICA 0.86, SCA 0.99) |
| Eine Elektrode (zwei Neuronen) | ✅ bei **sehr verschieden breiten** Spitzen (Ähnlichkeit 2) **0.97** gegen 0.66 für die Schwelle (ICA, SOBI 0.33, SCA 0.04); ❌ bei den gewohnten Breiten **0.50**, unter der Schwelle (0.66) |
| Rechenzeit | ✅ 0.4 s für die NMF im Standardfall (0.6 s für die ganze Analyse mit ICA, SOBI, SCA); 24 s im Extremfall (5 Neuronen, 8 Elektroden, 40000 Abtastwerte, 1000 Iterationen, unbekannte Komponentenzahl) |

## Was die Demo zeigt

1. **NMF in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Signal** → **Gleichrichten** (bei einer Elektrode: **Spektrogramm**) → **Faktorisieren** (Fehler je Iteration; Korrelation, Mischspalten-Kosinus und Spitzen-F1 an Zwischenständen – sie steigen schnell und können wieder fallen, obwohl der Fehler weiter sinkt) →
   **Komponenten** (Aktivitäten gegen die wahren Neuronen, Mischspalten gegen die wahren bzw. bei einer Elektrode die Spektren) → **Ergebnis** (Raster, Korrelationsmatrix).
2. **Was die NMF gefunden hat – im Vergleich mit ICA, SOBI und SCA:** Spitzen-F1 (für alle Verfahren dieselbe Definition, Zuordnung per Korrelation), Korrelation, Mischspalten-Kosinus, Fehler gegen den Fehler mit den wahren Mischspalten; Raster; Balken mit PCA und der besten Einzelelektrode; Urteil
   (Codes: falsche Komponentenzahl → eine Elektrode gut / schwach → NMF besser → zu wenige Elektroden → Rauschen → ein anderes Verfahren besser → gleichauf).
3. **📐 Sweeps** über Elektroden, Neuronen, Rauschen, Feuerrate, Ähnlichkeit, Schwankung, Synchronität, Rauschschwelle, Sparsität und Iterationen (feste Datensätze ab 100000, Streuung; ICA, SOBI, SCA in jedem Diagramm der Datenregler; dazu der Fehler der NMF gegen den mit den wahren Mischspalten).
4. **🔬 Komponentenzahl** (Fehlerkurven für 2–5 Neuronen, Trefferzahl des Knicks), **🔬 Initialisierung** (acht Zufallsstarts und NNDSVD, Fehler gegen Spitzen-F1) und **🧩 acht Szenen** – drei Stärken, fünf Schwächen der NMF (Experimente auf Abruf).
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Nicht-Negativität, Eindeutigkeit, Komponentenzahl, Rang, Unabhängigkeit, ein Kanal).

Regler: Neuronen (2–5), Elektroden (1–8), Feuerrate (×0.25–×4), Wellenform-Ähnlichkeit (0–2), Amplitudenschwankung (0–0.3), Synchronität (0–0.9), Rauschen, Länge, Gleichrichtung und Rauschschwelle (bei einer Elektrode ausgeblendet, Werte bleiben erhalten),
Komponentenzahl (bekannt / Knick), Initialisierung (zufällig / NNDSVD), Sparsität λ (0–3), Iterationen (20–1000).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | NMF | ICA | SOBI | SCA | Einzelelektrode | Schwelle | Urteil |
|---|---|---|---|---|---|---|---|
| Vier Neuronen, vier Elektroden | 0.73 | 1.00 | 1.00 | 1.00 | 0.65 | 0.27 | ein anderes Verfahren besser |
| Synchrones Feuern (90 %) | 0.99 | 0.62 | 0.81 | 0.81 | 0.91 | 0.86 | NMF besser |
| Starkes Rauschen (1.0, Schwelle 3) | 0.74 | 0.45 | 0.38 | 0.85 | 0.71 | 0.27 | ein anderes Verfahren besser |
| Eine Elektrode, sehr verschiedene Breiten | 0.89 | 0.31 | 0.31 | 0.05 | 0.31 | 0.65 | eine Elektrode: gut |
| Eine Elektrode, Breiten wie gewohnt | 0.61 | 0.31 | 0.31 | 0.05 | 0.31 | 0.65 | eine Elektrode: nicht besser als die Schwelle |
| Fünf Neuronen, Komponentenzahl unbekannt | 0.51 (k = 4) | 0.86 | 0.74 | 0.99 | 0.48 | 0.32 | falsche Komponentenzahl |

## Modell und Verfahren

- **Szenario** (`nm_scenario.py`): wie in den Vorgänger-Demos (10 kHz; Neuronen mit biphasischer Wellenform und Poisson-artigem Feuern mit 2 ms Refraktärzeit, Mischung ∝ 1/(d² + ε), Rauschen relativ zum Neuronen-Signal, Wellenform-Ähnlichkeit, Amplitudenschwankung); bei den Standardwerten bit-identisch (per Test).
  Neu: **Synchronität** s – der Anteil der Spikes, den die Neuronen 2 und weitere mit Neuron 1 teilen (bei 0 unverändert).
- **Nicht-negative Darstellung** (`nm_algorithm.py`): je Elektrode V = max(−(x − Median) − c·σ, 0) (nur die negative Spitze) oder max(|x − Median| − c·σ, 0) (Betrag), σ = MAD/0.6745, c = Rauschschwelle; bei **einer** Elektrode das Betragsspektrogramm (Hann-Fenster 32, Sprung 4), Aktivitäten je Rahmen werden auf das Abtastraster interpoliert.
- **NMF** (numpy von Grund auf): multiplikative Updates nach Lee und Seung für den Frobenius-Fehler, Spalten von W auf Summe 1 (Skala in H), optional eine **L1-Strafe** λ auf H; Abbruch bei relativer Fehleränderung < 10⁻⁵ oder Höchstzahl der Iterationen; Fehlerverlauf über die Spurformel (‖V‖² − 2·tr(WᵀVHᵀ) + tr(WᵀW·HHᵀ)) ohne die große Produktmatrix.
  Start: **zufällig** (fünf Neustarts, der mit dem kleinsten Fehler zählt) oder **NNDSVD** (deterministisch, gegen die Struktur bei scikit-learn geprüft). **Komponentenzahl:** bekannt oder der Knick der Fehlerkurve (kleinstes k ≥ 2, bei dem ein weiteres Element weniger als ein Viertel dessen spart, was das letzte brachte).
- **Vergleich:** ICA, SOBI und SCA wortgleich aus den Vorgänger-Demos (`nm_ica.py`, `nm_sobi.py`, `nm_sca.py`): Spitzen-F1 auf der geschätzten Neuronen-Spur – dieselbe Definition wie für die NMF (Zuordnung per Korrelation, Vorzeichen und Skala per Regression, Spitzen tiefste zuerst, Treffer ±4 Abtastwerte).
- **Auswertung** (`nm_evaluation.py`): Zuordnung (Bitmasken-DP), Spitzen-F1, Korrelation, **Mischspalten-Kosinus** (auch für ICA, SOBI, SCA aus der geschätzten Mischmatrix), **Vergleichsfehler** ‖V − A_wahr·H*‖/‖V‖ (H* per Updates bei festem W = A), Sweeps, Rang-, Initialisierungs- und Szenen-Tabellen, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Die NMF trennt im Standardfall schlechter als die drei Vorgänger** (0.77 gegen 1.00) – obwohl sie die Daten besser anpasst als die wahre Zerlegung. Vor dem Bau war die Vermutung, dass sie mit dem Sparsitäts-Argument (wenig gleichzeitige Spikes) ähnlich gut wie SCA trennt; gemessen: nein. Die positiven Mischspalten der vier Neuronen liegen dicht beieinander (Kosinus bis 0.90), und Nicht-Negativität allein legt die Zerlegung nicht eindeutig fest.
- **Mehr Iterationen machen es schlechter, nicht besser** (0.80 bei 100, 0.74 bei 1000 Iterationen, obwohl der Fehler weiter fällt): eine Zerlegung, die *noch* besser passt, ist nicht näher an der Wahrheit. Der Fehler allein ist kein Gütemaß für die Trennung.
- **Der Vorteil bei abhängigen Quellen hat eine Einschränkung:** bei 90 % gemeinsamen Spikes gewinnt die NMF (0.98 gegen 0.66 / 0.80 / 0.72), aber die beste Einzelelektrode erreicht dort 0.92 – bei so viel Gleichzeitigkeit ist die Aufgabe leicht und die Neuronen sind kaum noch zu unterscheiden. Es ist ein Sieg gegen Verfahren, deren Annahme verletzt ist, kein Beleg für sehr gute Trennung.
- **Einkanal funktioniert nur mit sehr verschiedenen Breiten:** bei den gewohnten Breiten (2 gegen 3 Abtastwerte) haben die Spitzen fast dieselben Spektren, die NMF liegt mit 0.50 unter der bloßen Schwelle (0.66); bei doppelt so großem Breitenunterschied (Ähnlichkeit 2: Breite 1 gegen 3) sind es 0.97. Bei vier Neuronen und einer Elektrode 0.38.
- **Der Knick der Fehlerkurve versagt bei fünf Neuronen** (immer vier gewählt): das fünfte Neuron senkt den Fehler zu wenig, um aufzufallen.
- **Sparsität hilft nur wenig** (0.77 → 0.81) und hebt den Fehler über den der Wahrheit; zufällige Neustarts streuen um etwa 0.03 im F1, NNDSVD ist mit 0.73 nicht besser.
- **Gleichrichtung ist nur eine Näherung:** Überlappungen addieren sich nur in der negativen Phase, der Nachschlag fällt weg (oder wird beim Betrag mitgezählt und verschmilzt mit der Spitze), Rauschen wird mitgleichgerichtet und braucht die Schwelle – bei starkem Rauschen ist sie die wichtigste Stellschraube (0.52 gegen 0.74).
- **Synthetische Daten:** feste Spitzenform je Neuron, exakt lineare positive Mischung, weißes Gauß'sches Rauschen, Elektroden auf einer Zeile. Literatur nur mit Namen: Lee und Seung (multiplikative Updates), Boutsidis und Gallopoulos (NNDSVD).

## Verifikation

- NMF: Fehler monoton nichtsteigend, Nicht-Negativität und Spaltennormierung erhalten; exakte Rückgewinnung einer rauschfreien trennbaren Handinstanz und eines Rang-1-Problems; Fehler auf einer Handinstanz **gegen scikit-learn** vergleichbar (< 0.02); NNDSVD-Struktur (erste Komponente = führendes Singulärpaar), Determinismus;
  Abbruchkriterium, Zwischenstände, Sparsitäts-Strafe, Neustarts; Gleichrichtung (Handinstanz, Rauschschwelle: Grundlinie ohne Schwelle, kaum mit), robuste Rauschschätzung; Spektrogramm eines reinen Tons (richtiger Bin); Interpolation; Knick der Fehlerkurve (Handinstanzen).
- Szenario bit-genau gegen die Vorgänger-Demos (Standardwerte), Synchronität 0 ändert nichts, > 70 % gemeinsame Spikes bei 0.9; ICA, SOBI, SCA und die Referenzen gegen eingefrorene Werte; Mischspalten-Kosinus, Spaltenähnlichkeit, Zuordnung, Spitzen-F1 mit Handinstanzen; Urteilscodes; Ansichten.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Neuronen, Elektroden, Feuerrate, Ähnlichkeit, Schwankung, Synchronität, Rauschen, Gleichrichtung, Schwelle, Sparsität, Iterationen, Initialisierung, Knick, Presets, Szenen, Grenzen-Tabelle; jeweils Mittel über die festen Sweep-Datensätze mit engen Toleranzen, positive **und** negative Aussagen);
  alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt mit einer und vier Elektroden, Randgrößen, ohne erkannte Spikes, ausgeblendete Gleichrichtungsregler behalten ihre Werte, Sweep-Optionen, Experimente auf Abruf), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Komponentenzahl und Initialisierung, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `nm_algorithm.py` | Gleichrichtung, Spektrogramm, NMF (multiplikative Updates), NNDSVD, Zufallsstart, Knick der Fehlerkurve |
| `nm_ica.py`, `nm_sobi.py`, `nm_sca.py` | Vergleichsverfahren (wortgleich aus ica/sobi/sca-demo) |
| `nm_scenario.py`, `nm_constants.py` | Mehrelektroden-Generator (mit Synchronität); Konstanten, Presets |
| `nm_evaluation.py` | Zuordnung, Kennzahlen, Mischspalten-Kosinus, Vergleich, Sweeps, Tabellen, Urteil |
| `nm_presets.py`, `nm_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | NMF (Handinstanzen, scikit-learn-Kreuzprüfung), Szenario, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
