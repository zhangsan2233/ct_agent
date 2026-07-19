from base64 import b64encode
from pathlib import Path

import streamlit as st


APPLE_WORKBENCH_CSS = """
<style>
:root {
    --ct-bg: #000000;
    --ct-surface: #1d1d1f;
    --ct-surface-raised: #242426;
    --ct-surface-muted: #0b0b0c;
    --ct-text: #f5f5f7;
    --ct-secondary: #86868b;
    --ct-tertiary: #6e6e73;
    --ct-separator: #424245;
    --ct-blue: #2997ff;
    --ct-cyan: #2997ff;
    --ct-positive: #ff453a;
    --ct-uncertain: #ff9f0a;
    --ct-negative: #30d158;
    --ct-radius: 8px;
}

html, body, [class*="css"] {
    font-family: "SF Pro Text", "SF Pro Display", -apple-system,
        BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Segoe UI",
        "Microsoft YaHei", sans-serif;
    color: var(--ct-text);
    letter-spacing: 0;
}

[data-testid="stAppViewContainer"] {
    background: var(--ct-bg);
}

[data-testid="stHeader"] {
    background: rgba(5, 5, 7, 0.72);
    border-bottom: 1px solid rgba(66, 66, 72, 0.6);
    backdrop-filter: blur(20px) saturate(150%);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1480px;
    padding: 0 2.25rem 4rem;
}

[data-testid="stSidebar"] {
    background: rgba(8, 8, 10, 0.96);
    border-right: 1px solid var(--ct-separator);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.25rem;
}

[data-testid="stSidebar"] h3 {
    color: var(--ct-text) !important;
}

.scan-hero {
    position: relative;
    height: 470px;
    margin: 0 -2.25rem 28px;
    overflow: hidden;
    background: #000000;
    border-bottom: 1px solid #25252a;
    isolation: isolate;
}

.scan-hero-image {
    position: absolute;
    top: 0;
    right: 0;
    width: 68%;
    height: 100%;
    object-fit: cover;
    object-position: 50% 50%;
    opacity: 0.88;
    filter: contrast(1.08) brightness(0.82);
}

.scan-hero-mask {
    position: absolute;
    inset: 0 auto 0 0;
    width: 48%;
    background: rgba(0, 0, 0, 0.96);
}

.scan-hero-content {
    position: absolute;
    z-index: 3;
    top: 112px;
    left: 54px;
    width: min(510px, 42%);
}

.scan-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #b8eaff;
    font-size: 11px;
    font-weight: 700;
    line-height: 1;
}

.scan-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ct-cyan);
    box-shadow: 0 0 12px rgba(70, 200, 255, 0.9);
}

.scan-hero-title {
    margin-top: 18px;
    color: #ffffff;
    font-size: 60px;
    font-weight: 720;
    line-height: 1.02;
}

.scan-hero-copy {
    max-width: 470px;
    margin-top: 18px;
    color: #d5d5da;
    font-size: 20px;
    font-weight: 500;
    line-height: 1.55;
}

.scan-hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    margin-top: 26px;
    color: #7f7f88;
    font-size: 10px;
    font-weight: 700;
}

.scan-reticle {
    position: absolute;
    z-index: 2;
    top: 50%;
    right: 24%;
    width: 176px;
    height: 176px;
    border: 1px solid rgba(70, 200, 255, 0.48);
    transform: translate(50%, -50%);
}

.scan-reticle::before,
.scan-reticle::after {
    content: "";
    position: absolute;
    background: rgba(70, 200, 255, 0.42);
}

.scan-reticle::before {
    top: 50%;
    left: -24px;
    width: calc(100% + 48px);
    height: 1px;
}

.scan-reticle::after {
    top: -24px;
    left: 50%;
    width: 1px;
    height: calc(100% + 48px);
}

.scan-beam {
    position: absolute;
    z-index: 2;
    left: 48%;
    right: 0;
    height: 1px;
    background: rgba(70, 200, 255, 0.9);
    box-shadow: 0 0 16px rgba(70, 200, 255, 0.85);
    animation: scan-pass 5.5s ease-in-out infinite;
}

.scan-coordinates {
    position: absolute;
    z-index: 3;
    right: 22px;
    bottom: 18px;
    color: rgba(216, 239, 249, 0.7);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 10px;
    line-height: 1.6;
    text-align: right;
}

@keyframes scan-pass {
    0%, 100% { top: 16%; opacity: 0.2; }
    50% { top: 84%; opacity: 0.95; }
}

h1, h2, h3, h4, p, label, button, input, textarea {
    letter-spacing: 0 !important;
}

h2, h3 {
    font-family: "SF Pro Display", "SF Pro Text", -apple-system,
        BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Segoe UI",
        sans-serif;
    color: var(--ct-text) !important;
    font-weight: 600 !important;
}

h3 {
    font-size: 20px !important;
    line-height: 1.25 !important;
    margin-top: 0.5rem !important;
}

p, li, label, [data-testid="stCaptionContainer"] {
    line-height: 1.55;
}

.workspace-kicker {
    margin: 8px 0 5px;
    color: var(--ct-cyan);
    font-size: 10px;
    font-weight: 750;
}

[data-testid="stForm"] {
    background: rgba(17, 17, 20, 0.9);
    border: 1px solid #333339;
    border-radius: var(--ct-radius);
    padding: 18px 18px 14px;
    box-shadow: 0 22px 80px rgba(0, 0, 0, 0.34);
    backdrop-filter: blur(20px) saturate(130%);
}

[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    color: var(--ct-text) !important;
    background: #09090b !important;
    border-color: #3a3a40 !important;
    border-radius: var(--ct-radius) !important;
}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: var(--ct-text) !important;
    caret-color: var(--ct-cyan);
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 96px;
    color: var(--ct-secondary);
    background: #09090b;
    border: 1px dashed #45454c;
    border-radius: var(--ct-radius);
}

[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button {
    min-height: 44px;
    border-radius: var(--ct-radius);
    font-weight: 650;
    transition: background-color 120ms ease, border-color 120ms ease, transform 120ms ease;
}

[data-testid="stButton"] button:active,
[data-testid="stFormSubmitButton"] button:active {
    transform: scale(0.985);
}

button[kind="primary"] {
    background: var(--ct-blue) !important;
    border-color: var(--ct-blue) !important;
    box-shadow: 0 0 22px rgba(10, 132, 255, 0.22);
}

button[kind="primary"]:hover {
    background: #2997ff !important;
    border-color: #2997ff !important;
}

[data-testid="stMetric"] {
    min-height: 102px;
    background: rgba(17, 17, 20, 0.9);
    border: 1px solid #34343a;
    border-radius: var(--ct-radius);
    padding: 15px 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--ct-secondary);
    font-size: 12px;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: var(--ct-text);
    font-size: 24px;
    font-weight: 680;
}

[data-testid="stAlert"],
[data-testid="stStatusWidget"],
[data-testid="stExpander"] {
    border-radius: var(--ct-radius);
}

[data-testid="stAlert"] {
    border-width: 1px;
}

[data-testid="stStatusWidget"] {
    background: var(--ct-surface);
    border: 1px solid var(--ct-separator);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    width: 100%;
    max-width: 100%;
    gap: 22px;
    padding: 0;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--ct-separator);
    border-radius: 0;
    overflow-x: auto;
}

[data-testid="stTabs"] [role="tab"] {
    min-height: 48px;
    padding: 0 2px;
    border-radius: 0;
    color: #9a9aa2;
    font-size: 15px;
    font-weight: 600;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #ffffff;
    background: transparent;
    box-shadow: inset 0 -2px 0 var(--ct-blue);
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--ct-separator);
    border-radius: var(--ct-radius);
    overflow: hidden;
}

[data-testid="stChatMessage"] {
    background: rgba(17, 17, 20, 0.92);
    border: 1px solid var(--ct-separator);
    border-radius: var(--ct-radius);
    padding: 12px 14px;
}

.input-provenance {
    display: flex;
    flex-wrap: wrap;
    gap: 7px 16px;
    margin: 12px 0 24px;
    padding: 10px 0;
    color: var(--ct-secondary);
    border-bottom: 1px solid var(--ct-separator);
    font-size: 12px;
}

.input-provenance strong {
    color: var(--ct-text);
    font-weight: 650;
}

.result-stage {
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(420px, 1.05fr);
    min-height: 440px;
    margin: 26px -2.25rem 0;
    overflow: hidden;
    background: #000000;
    border-top: 1px solid #25252a;
    border-bottom: 1px solid #25252a;
}

.scan-hero,
.result-stage,
.result-facts,
.result-review-note,
.agent-observatory,
.agent-detail-heading,
.scan-hero *,
.result-stage *,
.result-facts *,
.result-review-note *,
.agent-observatory *,
.agent-detail-heading * {
    font-family: "SF Pro Text", "SF Pro Display", -apple-system,
        BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Segoe UI",
        "Microsoft YaHei", sans-serif !important;
}

.scan-hero-title,
.result-stage-copy h2,
.observatory-header h3 {
    font-family: "SF Pro Display", "SF Pro Text", -apple-system,
        BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Segoe UI",
        "Microsoft YaHei", sans-serif !important;
}

.phase-number,
.event-index,
.scan-coordinates,
.result-image-meta,
.result-image-meta * {
    font-family: "SFMono-Regular", "SF Mono", Consolas, monospace !important;
}

.result-stage-copy {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
    padding: 48px 42px 50px 54px;
    background: #000000;
}

.result-eyebrow,
.observatory-kicker,
.agent-detail-heading span,
.execution-stream-heading span {
    color: var(--ct-blue);
    font-size: 14px;
    font-weight: 600;
}

.result-stage-copy h2 {
    max-width: 620px;
    margin: 14px 0 16px;
    color: #ffffff !important;
    font-size: 56px !important;
    font-weight: 720 !important;
    line-height: 1.05 !important;
    overflow-wrap: anywhere;
}

.result-stage-copy > p {
    max-width: 590px;
    margin: 0;
    color: #c7c7cd;
    font-size: 16px;
    line-height: 1.7;
}

.result-confidence {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 26px;
}

.result-confidence > strong {
    color: #ffffff;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}

.result-confidence > span {
    color: var(--ct-secondary);
    font-size: 14px;
    line-height: 1.35;
}

.result-confidence small {
    color: var(--ct-tertiary);
    font-size: 12px;
}

.result-gate {
    display: inline-flex;
    align-items: center;
    align-self: flex-start;
    gap: 8px;
    margin-top: 22px;
    color: var(--ct-secondary);
    font-size: 14px;
    font-weight: 600;
}

.result-gate span,
.observatory-state span,
.phase-status-dot {
    width: 7px;
    height: 7px;
    flex: 0 0 7px;
    border-radius: 50%;
    background: var(--ct-negative);
    box-shadow: 0 0 12px rgba(48, 209, 88, 0.72);
}

.result-gate.review span,
.workflow-phase.attention .phase-status-dot {
    background: var(--ct-uncertain);
    box-shadow: 0 0 12px rgba(255, 159, 10, 0.72);
}

.result-stage-media {
    position: relative;
    min-width: 0;
    min-height: 440px;
    overflow: hidden;
    background: #030304;
}

.result-stage-media img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    opacity: 0.9;
    filter: contrast(1.08) brightness(0.86);
}

.result-media-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    color: #52525a;
    font-size: 12px;
    font-weight: 700;
}

.result-reticle {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 172px;
    height: 172px;
    border: 1px solid rgba(70, 200, 255, 0.48);
    transform: translate(-50%, -50%);
}

.result-reticle::before,
.result-reticle::after {
    content: "";
    position: absolute;
    background: rgba(70, 200, 255, 0.4);
}

.result-reticle::before {
    top: 50%;
    left: -28px;
    width: calc(100% + 56px);
    height: 1px;
}

.result-reticle::after {
    top: -28px;
    left: 50%;
    width: 1px;
    height: calc(100% + 56px);
}

.result-scan-line {
    position: absolute;
    left: 0;
    right: 0;
    height: 1px;
    background: rgba(70, 200, 255, 0.84);
    box-shadow: 0 0 14px rgba(70, 200, 255, 0.72);
    animation: result-scan 5.8s ease-in-out infinite;
}

.result-image-meta {
    position: absolute;
    right: 18px;
    bottom: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
    color: rgba(222, 241, 249, 0.72);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 9px;
}

.result-image-meta strong {
    color: rgba(255, 255, 255, 0.84);
    font-size: 10px;
    font-weight: 600;
}

@keyframes result-scan {
    0%, 100% { top: 12%; opacity: 0.22; }
    50% { top: 88%; opacity: 0.92; }
}

.result-facts {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 0 -2.25rem;
    padding: 0 42px;
    background: #09090b;
    border-bottom: 1px solid var(--ct-separator);
}

.result-facts > div {
    min-width: 0;
    padding: 18px 12px;
    border-right: 1px solid var(--ct-separator);
}

.result-facts > div:last-child { border-right: 0; }

.result-facts span,
.workflow-telemetry span {
    display: block;
    margin-bottom: 5px;
    color: var(--ct-tertiary);
    font-size: 11px;
    font-weight: 600;
}

.result-facts strong {
    display: block;
    min-width: 0;
    color: var(--ct-text);
    font-size: 15px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.result-review-note {
    display: grid;
    grid-template-columns: 120px minmax(0, 1fr);
    gap: 20px;
    margin: 18px 0 8px;
    padding: 14px 0;
    border-top: 1px solid rgba(255, 159, 10, 0.48);
    border-bottom: 1px solid rgba(255, 159, 10, 0.24);
}

.result-review-note span {
    color: var(--ct-uncertain);
    font-size: 12px;
    font-weight: 600;
}

.result-review-note p {
    margin: 0;
    color: #c8c8ce;
    font-size: 14px;
}

.finding-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 28px;
    margin: 12px 0 20px;
}

.finding-group {
    min-width: 0;
    padding-top: 11px;
    border-top: 3px solid var(--ct-separator);
}

.finding-group.positive { border-top-color: var(--ct-positive); }
.finding-group.uncertain { border-top-color: var(--ct-uncertain); }

.finding-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 3px;
}

.finding-heading strong { font-size: 15px; }
.finding-heading span { color: var(--ct-secondary); font-size: 12px; }

.finding-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 12px;
    min-height: 44px;
    border-bottom: 1px solid var(--ct-separator);
}

.finding-name {
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: 14px;
    font-weight: 600;
}

.finding-score {
    color: var(--ct-secondary);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
}

.empty-finding {
    min-height: 44px;
    display: flex;
    align-items: center;
    color: var(--ct-secondary);
    border-bottom: 1px solid var(--ct-separator);
    font-size: 13px;
}

.section-note {
    color: var(--ct-secondary);
    font-size: 13px;
    line-height: 1.55;
    margin: -2px 0 16px;
}

.agent-route {
    margin: 8px 0 18px;
    padding: 11px 0;
    color: #8bdcff;
    border-top: 1px solid var(--ct-separator);
    border-bottom: 1px solid var(--ct-separator);
    font-size: 12px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.agent-observatory {
    margin: 24px 0 52px;
    padding: 52px 0 0;
    border-top: 1px solid var(--ct-separator);
}

.observatory-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 44px;
}

.observatory-header h3 {
    max-width: 860px;
    margin: 14px 0 0 !important;
    color: #ffffff !important;
    font-size: 48px !important;
    font-weight: 600 !important;
    line-height: 1.08 !important;
}

.observatory-state {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 4px;
    color: var(--ct-secondary);
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
}

.workflow-phases {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    border-top: 1px solid #3a3a41;
    border-bottom: 1px solid var(--ct-separator);
}

.workflow-phase {
    position: relative;
    min-width: 0;
    min-height: 156px;
    padding: 24px 24px 22px;
    border-right: 1px solid var(--ct-separator);
}

.workflow-phase:nth-child(3n) { border-right: 0; }
.workflow-phase:nth-child(-n + 3) { border-bottom: 1px solid var(--ct-separator); }

.workflow-phase::before {
    content: "";
    position: absolute;
    top: -1px;
    left: 0;
    width: 100%;
    height: 2px;
    background: var(--ct-cyan);
}

.workflow-phase.attention::before { background: var(--ct-uncertain); }

.phase-number {
    display: inline-block;
    margin-right: 7px;
    color: var(--ct-tertiary);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 12px;
}

.phase-status-dot { display: inline-block; }

.workflow-phase strong {
    display: block;
    margin-top: 24px;
    color: var(--ct-text);
    font-size: 20px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.workflow-phase small {
    display: block;
    margin-top: 10px;
    color: var(--ct-secondary);
    font-size: 13px;
    line-height: 1.5;
}

.workflow-telemetry {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 28px 0 52px;
    padding: 0;
    background: #0b0b0c;
    border-top: 1px solid var(--ct-separator);
    border-bottom: 1px solid var(--ct-separator);
}

.workflow-telemetry > div {
    min-width: 0;
    min-height: 92px;
    padding: 21px 24px;
    border-right: 1px solid var(--ct-separator);
    border-bottom: 1px solid var(--ct-separator);
}

.workflow-telemetry > div:nth-child(3n) { border-right: 0; }
.workflow-telemetry > div:nth-last-child(-n + 3) { border-bottom: 0; }

.workflow-telemetry strong {
    display: block;
    color: var(--ct-text);
    font-size: 19px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.execution-stream-heading,
.agent-detail-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin: 28px 0 18px;
}

.execution-stream-heading strong,
.agent-detail-heading strong {
    color: var(--ct-secondary);
    font-size: 14px;
    font-weight: 600;
}

.workflow-events {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0;
    border-top: 1px solid var(--ct-separator);
}

.workflow-event {
    display: grid;
    grid-template-columns: 64px minmax(0, 1fr);
    min-width: 0;
    padding: 22px 8px;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--ct-separator);
    border-radius: 0;
}

.workflow-event.attention { border-bottom-color: rgba(255, 159, 10, 0.52); }

.event-index {
    color: var(--ct-secondary);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 13px;
}

.event-copy { min-width: 0; }

.event-copy > div {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
}

.event-copy strong {
    min-width: 0;
    color: var(--ct-text);
    font-size: 20px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.event-copy > div span {
    color: var(--ct-negative);
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}

.workflow-event.attention .event-copy > div span { color: var(--ct-uncertain); }

.event-copy p {
    margin: 9px 0 10px;
    color: #b0b0b5;
    font-size: 15px;
    line-height: 1.55;
}

.event-copy small {
    display: block;
    color: var(--ct-secondary);
    font-size: 13px;
    overflow-wrap: anywhere;
}

@keyframes apple-content-reveal {
    from {
        opacity: 0;
        transform: translateY(36px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@supports (animation-timeline: view()) {
    .observatory-header,
    .workflow-phase,
    .workflow-telemetry,
    .workflow-event,
    .agent-detail-heading {
        animation: apple-content-reveal linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 24%;
    }
}

@media (prefers-reduced-motion: reduce) {
    .scan-beam { animation: none; top: 50%; }
    .result-scan-line { animation: none; top: 50%; }
    .observatory-header,
    .workflow-phase,
    .workflow-telemetry,
    .workflow-event,
    .agent-detail-heading {
        animation: none;
        opacity: 1;
        transform: none;
    }
}

@media (max-width: 820px) {
    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        max-width: 100vw;
        overflow-x: hidden;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 0 1rem 3rem;
    }

    .scan-hero {
        height: 440px;
        margin: 0 -1rem 22px;
    }

    .scan-hero-image {
        width: 100%;
        opacity: 0.53;
        object-position: 58% 50%;
    }

    .scan-hero-mask {
        width: 100%;
        background: rgba(0, 0, 0, 0.5);
    }

    .scan-hero-content {
        top: 92px;
        left: 24px;
        width: calc(100% - 48px);
    }

    .scan-hero-title { font-size: 38px; }
    .scan-hero-copy { font-size: 17px; max-width: 330px; }
    .scan-hero-meta {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 5px;
        margin-top: 20px;
        font-size: 9px;
        line-height: 1.25;
    }

    .scan-hero-meta span {
        display: block;
        max-width: 100%;
        overflow-wrap: anywhere;
    }
    .scan-reticle { right: 35%; width: 132px; height: 132px; opacity: 0.65; }
    .scan-coordinates { display: none; }

    .result-stage {
        grid-template-columns: minmax(0, 1fr);
        margin: 20px -1rem 0;
    }

    .result-stage-copy {
        min-height: 390px;
        padding: 42px 24px 36px;
    }

    .result-stage-copy h2 {
        font-size: 42px !important;
    }

    .result-stage-copy > p { font-size: 15px; }
    .result-stage-media { min-height: 320px; }
    .result-reticle { width: 132px; height: 132px; }

    .result-facts {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin: 0 -1rem;
        padding: 0 12px;
    }

    .result-facts > div:nth-child(2) { border-right: 0; }
    .result-facts > div:nth-child(-n + 2) { border-bottom: 1px solid var(--ct-separator); }

    .result-review-note {
        grid-template-columns: minmax(0, 1fr);
        gap: 6px;
    }

    .observatory-header {
        align-items: flex-start;
        flex-direction: column;
        gap: 12px;
    }

    .observatory-header h3 { font-size: 32px !important; }

    .workflow-phases {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .workflow-phase:nth-child(2n) { border-right: 0; }
    .workflow-phase {
        min-height: 146px;
        padding: 22px 16px 20px;
        border-bottom: 1px solid var(--ct-separator);
    }

    .workflow-phase:nth-child(3n) { border-right: 1px solid var(--ct-separator); }
    .workflow-phase:nth-child(2n) { border-right: 0; }
    .workflow-phase strong { font-size: 18px; }
    .workflow-phase small { font-size: 12px; }

    .workflow-telemetry {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .workflow-telemetry > div {
        min-height: 86px;
        padding: 18px 14px;
        border-bottom: 1px solid var(--ct-separator);
    }

    .workflow-telemetry > div:nth-child(3n) { border-right: 1px solid var(--ct-separator); }
    .workflow-telemetry > div:nth-child(2n) { border-right: 0; }
    .workflow-telemetry > div:nth-last-child(-n + 2) { border-bottom: 0; }

    .workflow-event {
        grid-template-columns: 42px minmax(0, 1fr);
        padding: 20px 0;
    }

    .event-copy > div {
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 18px; }
    [data-testid="stTabs"] [role="tab"] { min-height: 44px; }

    .finding-grid {
        grid-template-columns: minmax(0, 1fr);
        gap: 20px;
    }

    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }

    [data-testid="column"] {
        min-width: min(100%, 260px) !important;
        flex: 1 1 260px !important;
    }
}
</style>
"""


@st.cache_data
def _hero_image_data_url() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "cases"
        / "valid_4_a_1"
        / "slice_105_lung.png"
    )
    if not path.exists():
        return ""
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def apply_ui_theme() -> None:
    st.markdown(APPLE_WORKBENCH_CSS, unsafe_allow_html=True)


def render_product_header() -> None:
    image_url = _hero_image_data_url()
    st.markdown(
        f"""
        <section class="scan-hero">
            <img class="scan-hero-image" src="{image_url}" alt="胸部 CT 肺窗轴位切片">
            <div class="scan-hero-mask" aria-hidden="true"></div>
            <div class="scan-beam" aria-hidden="true"></div>
            <div class="scan-reticle" aria-hidden="true"></div>
            <div class="scan-hero-content">
                <div class="scan-status">
                    <span class="scan-status-dot"></span>
                    SYSTEM READY · MULTIMODAL EVIDENCE
                </div>
                <div class="scan-hero-title">ChestCT Agent</div>
                <div class="scan-hero-copy">
                    让每一条结论，都能追溯到影像、报告与检索证据。
                </div>
                <div class="scan-hero-meta">
                    <span>3D CHEST CT</span>
                    <span>AGENTIC RAG</span>
                    <span>MULTIMODAL FUSION</span>
                </div>
            </div>
            <div class="scan-coordinates">
                AXIAL · LUNG WINDOW<br>
                SLICE 105 / 212<br>
                DE-IDENTIFIED DEMO VOLUME
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
