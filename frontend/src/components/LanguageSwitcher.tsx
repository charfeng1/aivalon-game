import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import i18n from "../i18n";

interface LanguageSwitcherProps {
  className?: string;
}

function LanguageSwitcher({ className = "" }: LanguageSwitcherProps) {
  const { t, i18n: i18nInstance } = useTranslation();

  useEffect(() => {
    const savedLang = localStorage.getItem("i18n.lang");
    if (savedLang && (savedLang === "zh-CN" || savedLang === "en-US")) {
      i18n.changeLanguage(savedLang);
    }
  }, []);

  const setLang = (lang: "zh-CN" | "en-US") => {
    i18n.changeLanguage(lang);
    localStorage.setItem("i18n.lang", lang);
  };

  const currentLang = i18nInstance.language;

  return (
    <div className={`flex gap-2 ${className}`}>
      <button
        onClick={() => setLang("zh-CN")}
        className={`px-2 py-1 text-sm rounded border ${currentLang === "zh-CN" ? "bg-blue-100 border-blue-300" : "border-gray-300 hover:bg-gray-100"}`}
      >
        {t("language.switcher.zh")}
      </button>
      <button
        onClick={() => setLang("en-US")}
        className={`px-2 py-1 text-sm rounded border ${currentLang === "en-US" ? "bg-blue-100 border-blue-300" : "border-gray-300 hover:bg-gray-100"}`}
      >
        {t("language.switcher.en")}
      </button>
    </div>
  );
}

export default LanguageSwitcher;