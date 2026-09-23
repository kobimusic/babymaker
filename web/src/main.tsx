import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Babymaker } from "./Babymaker.tsx";
import "./tokens.css";

createRoot(document.getElementById("demo")!).render(
  <StrictMode>
    <Babymaker />
  </StrictMode>,
);
