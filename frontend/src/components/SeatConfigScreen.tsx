import { useState } from "react";
import type { ProviderConfig, SeatConfig } from "../types";
import "../styles/SeatConfigScreen.css";

interface SeatConfigScreenProps {
  providers: ProviderConfig[];
  onStart: (config: Record<number, SeatConfig>) => void;
  onCancel: () => void;
  t: (key: string) => string;
}

function SeatConfigScreen({ providers, onStart, onCancel, t }: SeatConfigScreenProps): JSX.Element {
  // Initialize with default config: all AI seats with first available provider
  const defaultProvider = providers[0]?.name || "";
  const defaultModel = providers[0]?.models[0] || "";

  const [numSeats, setNumSeats] = useState<number>(5);
  const [seatConfig, setSeatConfig] = useState<Record<number, SeatConfig>>({
    1: { isHuman: false, provider: defaultProvider, model: defaultModel },
    2: { isHuman: false, provider: defaultProvider, model: defaultModel },
    3: { isHuman: true },  // Default human seat
    4: { isHuman: false, provider: defaultProvider, model: defaultModel },
    5: { isHuman: false, provider: defaultProvider, model: defaultModel },
  });

  const toggleSeatHuman = (seat: number) => {
    const currentConfig = seatConfig[seat];
    const isCurrentlyHuman = currentConfig?.isHuman;

    if (isCurrentlyHuman) {
      // Deselect human - convert to AI
      const provider = providers.find((p) => p.name === defaultProvider);
      const model = provider?.models[0] || "";

      setSeatConfig((prev) => ({
        ...prev,
        [seat]: {
          isHuman: false,
          provider: defaultProvider,
          model: model,
        },
      }));
    } else {
      // Set this seat as human, all others as AI
      const newConfig: Record<number, SeatConfig> = {};
      for (let i = 1; i <= numSeats; i++) {
        if (i === seat) {
          newConfig[i] = { isHuman: true };
        } else {
          const existingConfig = seatConfig[i];
          const existingProvider = existingConfig?.provider || defaultProvider;
          const provider = providers.find((p) => p.name === existingProvider);
          const model = existingConfig?.model || provider?.models[0] || "";

          newConfig[i] = {
            isHuman: false,
            provider: existingProvider,
            model: model,
          };
        }
      }
      setSeatConfig(newConfig);
    }
  };

  const setSeatProvider = (seat: number, providerName: string) => {
    // When changing provider, automatically select the first model from that provider
    const provider = providers.find((p) => p.name === providerName);
    const firstModel = provider?.models[0] || "";

    setSeatConfig((prev) => ({
      ...prev,
      [seat]: {
        ...prev[seat],
        provider: providerName,
        model: firstModel,
      },
    }));
  };

  const setSeatModel = (seat: number, modelId: string) => {
    setSeatConfig((prev) => ({
      ...prev,
      [seat]: {
        ...prev[seat],
        model: modelId,
      },
    }));
  };

  const addSeat = () => {
    if (numSeats < 10) {
      const newSeatNumber = numSeats + 1;
      const provider = providers.find((p) => p.name === defaultProvider);
      const model = provider?.models[0] || "";

      setNumSeats(newSeatNumber);
      setSeatConfig((prev) => ({
        ...prev,
        [newSeatNumber]: {
          isHuman: false,
          provider: defaultProvider,
          model: model,
        },
      }));
    }
  };

  const removeSeat = () => {
    if (numSeats > 5) {
      const newConfig = { ...seatConfig };
      delete newConfig[numSeats];
      setSeatConfig(newConfig);
      setNumSeats(numSeats - 1);
    }
  };

  const isConfigValid = (): boolean => {
    // Check that all seats are configured
    for (let seat = 1; seat <= numSeats; seat++) {
      const config = seatConfig[seat];
      if (!config) return false;
      if (!config.isHuman && (!config.model || !config.provider)) return false;
    }
    return true;
  };

  const handleStart = () => {
    if (isConfigValid()) {
      onStart(seatConfig);
    }
  };

  return (
    <div className="seat-config-backdrop">
      <div className="seat-config-modal">
        <h2>{t("config.title")}</h2>
        <p className="config-instructions">{t("config.instructions")}</p>

        <div className="player-count-controls">
          <span>{t("config.numPlayers")}: {numSeats}</span>
          <button type="button" onClick={removeSeat} disabled={numSeats <= 5}>
            −
          </button>
          <button type="button" onClick={addSeat} disabled={numSeats >= 10}>
            +
          </button>
        </div>

        <div className="seat-config-grid">
          {Array.from({ length: numSeats }, (_, i) => i + 1).map((seat) => {
            const config = seatConfig[seat];
            const isHuman = config?.isHuman || false;

            return (
              <div key={seat} className={`seat-card ${isHuman ? "human-seat" : "ai-seat"}`}>
                <div className="seat-header">
                  <h3>{t("config.seat")} {seat}</h3>
                  <button
                    type="button"
                    className={`seat-type-toggle ${isHuman ? "human" : "ai"}`}
                    onClick={() => toggleSeatHuman(seat)}
                  >
                    {isHuman ? "👤 " + t("config.human") : "🤖 " + t("config.ai")}
                  </button>
                </div>

                {!isHuman && (
                  <div className="model-selector">
                    <label htmlFor={`seat-${seat}-provider`}>{t("config.provider")}</label>
                    <select
                      id={`seat-${seat}-provider`}
                      value={config?.provider || ""}
                      onChange={(e) => setSeatProvider(seat, e.target.value)}
                    >
                      {providers.map((provider) => (
                        <option key={provider.name} value={provider.name}>
                          {provider.display_name}
                        </option>
                      ))}
                    </select>

                    <label htmlFor={`seat-${seat}-model`}>{t("config.model")}</label>
                    <select
                      id={`seat-${seat}-model`}
                      value={config?.model || ""}
                      onChange={(e) => setSeatModel(seat, e.target.value)}
                      disabled={!config?.provider}
                    >
                      {config?.provider &&
                        providers
                          .find((p) => p.name === config.provider)
                          ?.models.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                    </select>
                  </div>
                )}

                {isHuman && (
                  <div className="human-indicator">
                    <span className="human-label">{t("config.youPlayHere")}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="config-actions">
          <button type="button" className="ghost" onClick={onCancel}>
            {t("button.cancel")}
          </button>
          <button
            type="button"
            className="primary"
            onClick={handleStart}
            disabled={!isConfigValid()}
          >
            {t("button.startGame")}
          </button>
        </div>
      </div>
    </div>
  );
}

export default SeatConfigScreen;