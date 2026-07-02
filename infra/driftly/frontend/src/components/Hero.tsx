// Static hero header (medi-watch README style). The live state lives in the
// Monitor view's verdict card, so the hero stays a calm front door.
export function Hero() {
  return (
    <section className="hero" id="hero">
      <div className="hero-eyebrow">drift monitoring · live</div>

      <h1>
        <span className="light">Model drift,</span>{" "}
        <span className="accent">monitored</span> &amp; simulated.
      </h1>

      <p className="hero-deck">
        <b>Driftly</b> watches the running dataset for distribution drift on three axes —{" "}
        <b>Wasserstein</b>, <b>PSI</b>, and <b>KL</b> — and lets you{" "}
        <b>inject drift scenarios</b> to exercise the retrain loop. Monitor shows the live
        state; Simulator drives it.
      </p>

      <div className="hero-facts">
        <div className="hero-fact"><span className="k">Monitor</span><span className="v">live drift of current.csv</span></div>
        <div className="hero-fact"><span className="k">Simulator</span><span className="v">inject → trigger Airflow</span></div>
        <div className="hero-fact"><span className="k">Metrics</span><span className="v">Wasserstein · PSI · KL</span></div>
        <div className="hero-fact"><span className="k">Reference</span><span className="v">features.csv · NB04</span></div>
      </div>
    </section>
  );
}
