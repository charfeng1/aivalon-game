import i18n from "i18next";
import { initReactI18next } from "react-i18next";

// Import translation files
import zhCN from "./locales/zh-CN.json";
import enUS from "./locales/en-US.json";

// Initialize i18next
i18n
  .use(initReactI18next)
  .init({
    resources: {
      "zh-CN": { translation: zhCN },
      "en-US": { translation: enUS }
    },
    lng: "zh-CN", // default language
    fallbackLng: "en-US",
    interpolation: {
      escapeValue: false // react already safes from xss
    }
  });

export default i18n;