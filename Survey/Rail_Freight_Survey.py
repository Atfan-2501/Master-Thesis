import streamlit as st
from firebase_config import initialize_firebase
import base64
import uuid
from datetime import datetime
import pandas as pd


# Initialize Firebase
db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    def init_firestore():
        global db
        if db is not None:
            return db
        if not firebase_admin._apps:
            # Replace escaped newlines if needed:
            sa = dict(st.secrets["firebase_credentials"])
            if "\n" in sa.get("private_key", ""):
                sa["private_key"] = sa["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(sa)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        return db
except Exception:
    # Allow app to run without firebase installed
    pass

# ---- THEME + GLOBAL CSS ----
def inject_css(
    *,
    card_bg="rgba(255,255,255,0.92)",
    card_radius="12px",
    card_pad="10px 12px",
    gap_below="10px",
    option_gap="6px",
    label_gap="6px",
    font_color="#000",
):
    st.markdown(f"""
    <style>
    :root {{
      --card-bg: {card_bg};
      --card-radius: {card_radius};
      --card-pad: {card_pad};
      --gap-below: {gap_below};
      --option-gap: {option_gap};
      --label-gap: {label_gap};
      --font-color: {font_color};
    }}

    /* Generic card */
    .card {{
      background: var(--card-bg);
      padding: var(--card-pad);
      border-radius: var(--card-radius);
      margin: 6px 0 var(--gap-below) 0;
      color: var(--font-color);
    }}

    /* Question label pill */
    .q-label {{
      background: var(--card-bg);
      padding: 6px 8px;
      border-radius: 8px;
      margin: 6px 0 6px 0;
      color: var(--font-color);
      font-weight: 800;
    }}

    /* QA container (label + widget in one box if you want) */
    .qa-card {{
      background: var(--card-bg);
      padding: var(--card-pad);
      border-radius: var(--card-radius);
      margin: 6px 0 var(--gap-below) 0;
    }}
    .qa-title {{
      color: var(--font-color);
      font-weight: 800;
      margin: 0 0 var(--label-gap) 0;
    }}

    /* Streamlit widgets inside */
    div[role="radiogroup"] {{
      display: flex; flex-wrap: wrap;
      gap: var(--option-gap);
      margin-top: 0;   /* no extra gap above options */
    }}
    div[role="radiogroup"] > label {{
      margin: 0 !important;
      padding: 2px 6px;
    }}

    /* Sliders look like cards and keep spacing tight */
    div[data-testid="stSlider"] {{
      background: var(--card-bg);
      padding: 6px;
      border-radius: 8px;
      margin: 6px 0 var(--gap-below) 0;
    }}

    /* Number inputs (importance matrix) */
    div[data-testid="stNumberInput"] label {{ margin-bottom: 2px; }}
    div[data-testid="stNumberInput"] > div {{ margin-top: 0; }}

    /* Matrix blocks */
    .matrix-label, .matrix-hint, .matrix-row {{
      background: var(--card-bg);
      border-radius: 8px;
      padding: 6px 8px;
      margin: 6px 0;
      color: var(--font-color);
    }}
    .matrix-label b {{ font-weight: 800; }}
    .matrix-hint i {{ opacity: .9; }}

    /* Markdown paragraph default tightening */
    div[data-testid="stMarkdownContainer"] p {{ margin: .25rem 0; }}

    /* Tables */
    table {{ margin-bottom: 8px; background: white !important}}
    th, td {{ padding: 6px 8px !important; color: var(--font-color); }}
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="CH Intermodal Freight Survey", layout="centered")

inject_css(card_bg="rgba(255,255,255,0.92)", option_gap="6px", label_gap="6px")

# ------------------------------
# UPDATED SP TASKS (4 attributes + Opt-out)
# ------------------------------

SP_COLUMNS = [
    "Cost (CHF/TEU)",
    "Transit time (h)",
    "On-time reliability (% within ±2h)",
    "Departures per day",
]

# ------------------------------
# NEW: Load blocked SP design from CSVs and assign block/respondent
# ------------------------------
from pathlib import Path

def load_sp_designs(core_path: str = "sp_core_design_blocks.csv", checks_path: str = "sp_checks_design.csv"):
    core_p = Path(core_path)
    checks_p = Path(checks_path)
    if not core_p.exists() or not checks_p.exists():
        return None, None
    core_df = pd.read_csv(core_p)
    checks_df = pd.read_csv(checks_p)
    return core_df, checks_df


def ensure_respondent_context():
    # respondent_id stable within session
    if "respondent_id" not in st.session_state:
        st.session_state.respondent_id = str(uuid.uuid4())
        st.session_state.started_at_utc = datetime.utcnow().isoformat() + "Z"
    # deterministic block assignment: UUID hash modulo 3 + 1
    if "assigned_block" not in st.session_state:
        h = abs(hash(st.session_state.respondent_id))
        st.session_state.assigned_block = (h % 3) + 1


def build_tasks_for_block(core_df: pd.DataFrame, checks_df: pd.DataFrame, block_id: int):
    # Core tasks for given block (1..8)
    core = core_df[core_df.block_id == block_id].copy()
    tasks = []
    for t in sorted(core.task_in_block.unique()):
        tt = core[core.task_in_block == t]
        road = tt[tt.alt_id == 'Road'].iloc[0]
        inter = tt[tt.alt_id == 'Intermodal'].iloc[0]
        alts = [
            {"alt": "Road", "Cost (CHF/TEU)": int(road.cost_chf_teu), "Transit time (h)": int(road.time_h),
             "On-time reliability (% within ±2h)": int(road.ontime_p), "Departures per day": int(road.dep_per_day)},
            {"alt": "Intermodal", "Cost (CHF/TEU)": int(inter.cost_chf_teu), "Transit time (h)": int(inter.time_h),
             "On-time reliability (% within ±2h)": int(inter.ontime_p), "Departures per day": int(inter.dep_per_day)},
        ]
        tasks.append((t, alts, False, ""))
    # Checks appended as tasks 9 and 10
    for t in sorted(checks_df.task_in_block.unique()):
        tt = checks_df[checks_df.task_in_block == t]
        r = tt[tt.alt_id == 'Road'].iloc[0]
        im = tt[tt.alt_id == 'Intermodal'].iloc[0]
        alts = [
            {"alt": "Road", "Cost (CHF/TEU)": int(r.cost_chf_teu), "Transit time (h)": int(r.time_h),
             "On-time reliability (% within ±2h)": int(r.ontime_p), "Departures per day": int(r.dep_per_day)},
            {"alt": "Intermodal", "Cost (CHF/TEU)": int(im.cost_chf_teu), "Transit time (h)": int(im.time_h),
             "On-time reliability (% within ±2h)": int(im.ontime_p), "Departures per day": int(im.dep_per_day)},
        ]
        tasks.append((8 + t, alts, True, str(tt.iloc[0].check_type)))
    return tasks


def save_long_rows_to_firebase(rows: list):
    try:
        client = init_firestore()
        batch_ok = True
        for r in rows:
            doc_ref = client.collection("ch_intermodal_sp_long").document()
            doc_ref.set(r)
        return True, "ok"
    except Exception as e:
        return False, str(e)

survey_schema = [
    {
        "section": "Section 1: Company Profile",
        "items": [
            {"id":"company_profile","label":"Company Profile (select all that apply)","type":"multiselect",
             "options":["Manufacturer","Retailer/Wholesaler","Freight Forwarder (3PL/4PL)","Intermodal Operator (Rail, Terminal)","Other"]},
            {"id":"industry_sector","label":"Industry Sector","type":"multiselect",
             "options":["Automotive","Chemicals","Consumer Goods","Industrial Equipment","Dangerous goods","Perishable goods","Food & Beverage","Other"]},
            {"id":"annual_teu","label":"Annual Freight Volume (TEU per year)","type":"radio",
             "options":["< 100","100 – 500","500 – 1,000","> 1,000"]},
            {"id":"cost_per_teu","label":"Transport Cost per TEU (CHF per TEU)","type":"radio",
             "options":["< 500","500 – 1,000","1,000 – 2,000","> 2,000"]},
            {"id":"shipment_types","label":"Primary shipment types","type":"multiselect",
             "options":["Full Trailer Load (FTL)","Containers","Semitrailers","Bulk Freight"]},
            {"id":"distance_ch","label":"Typical transport distance within Switzerland","type":"radio",
             "options":["< 100 km","100–200 km","200–350 km","> 350 km"]},
            {"id":"mode_decider","label":"Who makes transport mode decisions?","type":"multiselect",
             "options":["Logistics / Supply Chain Manager","CEO / Senior Executive","Operations Manager","Procurement / Purchasing","Other"]},
        ],
    },
    {
        "section": "Section 2: Current Transport Mode & Segmentation",
        "items": [
            {"id":"existing_mode","label":"Primary transport mode in Switzerland","type":"radio",
             "options":["Road","Multimodal","Other"]},
            {"id":"secondary_modes","label":"Other modes used occasionally (select all that apply)","type":"multiselect",
             "options":["Road","Multimodal","None"]},

            {"id":"reasons_current","label":"Reasons for current mode (select all that apply)","type":"multiselect",
             "options":["Cost efficiency","Transit time / Speed","Reliability (on-time)","Flexibility","Accessibility (terminal/first-last mile)","Sustainability (CO₂)","Risk avoidance","Regulatory compliance","Technological integration (tracking, digital booking)"]},
            {"id":"use_intermodal_12m","label":"Used intermodal rail in last 12 months?","type":"radio","options":["Yes","No"]},
            {"id":"intermodal_frequency","label":"If yes, how often do you use intermodal?","type":"radio","options":["Occasionally (1–5/yr)","Regularly (6+/yr)","Always"]},

            {"id":"nonuser_reasons","label":"If NOT using intermodal, why?","type":"multiselect",
             "options":["Cost is too high","Rail schedules not flexible","Transit time too long","Terminal access inconvenient","Need last-mile road","Complicated booking","Lack of tracking","Delays","Damage/loss","Other"]},
            {"id":"stop_using_reasons","label":"If no longer using intermodal, reasons","type":"multiselect",
             "options":["High costs","Limited rail network coverage","Long transit time","Inflexible schedules","Damage/loss concerns","Complex coordination with rail operators","Other"]},
            {"id":"user_reasons","label":"If using intermodal, key reasons","type":"multiselect",
             "options":["Cost savings","Appropriate transport time","Customised service","Environmental benefits","Improved reliability","Other"]},
        ],
    },
    {
        "section": "Section 3: Mode Choice Factors & Preferences",
        "items": [
            {"id":"factor_importance","label":"Rate importance (1–5) for each factor","type":"matrix_likert5",
             "rows":[
                "Cost","Transport Time","Service Frequency","Punctuality (On-Time)",
                "Terminal Accessibility","CO₂ Emissions / Sustainability",
                "Flexibility (Schedule)","Cargo Security & Damage Risk",
                "Digital Tracking","Booking Convenience"
             ]},
            {"id":"improvements","label":"What improvements would increase intermodal usage?","type":"multiselect",
             "options":["Lower costs","Faster transport times","More frequent rail schedules","Better reliability","Improved terminal access","Digital tracking solutions","Easy booking","Improved transparency","Other"]},
        ],
    },
    {
        "section": "Section 4: Psychological / Behavioral Factors",
        "items": [
            {"id":"trust_overall","label":"Trust in intermodal rail vs trucking (1–5)","type":"likert5"},
            {"id":"on_time_perf","label":"For users: rail meets scheduled delivery times (1–5)","type":"likert5"},
            {"id":"delay_severity_single","label":"How serious are delays? (1–5)","type":"likert5"},
            {"id":"flexibility_vs_truck","label":"Flexibility vs truck (1–5)","type":"likert5"},
            {"id":"service_frequency_fit","label":"Adequacy of service frequency (1–5)","type":"likert5"},
            {"id":"delay_severity_table","label":"Delay severity by duration","type":"matrix_ordinal",
             "rows":["1 Day","2 Days","3 Days","4 Days","5+ Days"],
             "cols":["Not Serious at All","Slightly Serious","Moderately Serious","Highly Serious","Very Serious"]},
            {"id":"cost_perception","label":"Cost of rail vs road (1=Much More Expensive, 5=Much Cheaper)","type":"likert5"},
            {"id":"risk_damage","label":"Risk of damage/theft/loss (1–5)","type":"likert5"},
            {"id":"industry_influence","label":"Influence of industry trends/competitors (1–5)","type":"likert5"},
            {"id":"sustainability_importance","label":"Importance of sustainability (1–5)","type":"likert5"},
            {"id":"low_carbon_priority","label":"Priority for low-carbon modes (1–5)","type":"likert5"},
            {"id":"pay_premium_co2","label":"Willingness to pay CO₂ premium (1–5)","type":"likert5"},
            {"id":"stick_current_mode","label":"Likelihood to keep current mode (1–5)","type":"likert5"},
            {"id":"time_pressure","label":"Decisions under time pressure (1–5)","type":"likert5"},
            {"id":"pressure_to_use_rail","label":"Felt pressure to use intermodal rail (1–5)","type":"likert5"},
            {"id":"extra_comm_time","label":"Extra communication time vs other modes (1–5)","type":"likert5"},
            {"id":"branch_specific_need","label":"Need for branch-specific services (1–5)","type":"likert5"},
            {"id":"finance_complexity","label":"Financial process complexity (1–5)","type":"likert5"},
            {"id":"transparency","label":"Process transparency (1–5)","type":"likert5"},
            {"id":"admin_complexity","label":"Administrative process complexity (1–5)","type":"likert5"},
            {"id":"portfolio_fit","label":"Service portfolio fit (1–5)","type":"likert5"},
            {"id":"terminal_access","label":"Terminal accessibility (1–5)","type":"likert5"},
            {"id":"meets_requirements","label":"Meets logistics requirements (1–5)","type":"likert5"},
            {"id":"booking_convenience","label":"Booking convenience in CH (1–5)","type":"likert5"},
            {"id":"digital_tracking_importance","label":"Importance of digital tracking (1–5)","type":"likert5"},
            {"id":"comfort_new_solutions","label":"Comfort with trying intermodal rail (1–5)","type":"likert5"},
            {"id":"concerns","label":"Top concerns about intermodal rail","type":"multiselect",
             "options":["Unreliable schedules","Poor customer service","Lack of flexibility","Terminal access issues","Higher cost","Other"]},
            {"id":"psych_open","label":"Other psychological/behavioral factors","type":"textarea"},
        ],
    },
    {
        "section": "Section 5: Environmental Impact",
        "items": [
            {"id":"co2_importance","label":"Importance of CO₂ reduction (1–5)","type":"likert5"},
            {"id":"sustainable_energy","label":"Importance of sustainable/alternative energy (1–5)","type":"likert5"},
        ],
    },
    {
        "section": "Section 6: Policy & Regulatory",
        "items": [
            {"id":"policy_encouragement","label":"Policies that would encourage Intermodal (select all)","type":"multiselect",
             "options":["Subsidies for rail transport","Carbon tax on road transport","Investment in rail infrastructure","Digital freight platforms for easier booking","More flexible rail schedules","Priority access for time-sensitive freight","Branch-specific intermodal services","Other"]},
            {"id":"aware_regulations","label":"Aware of Swiss transport regulations impacting mode choice?","type":"radio","options":["Yes","No"]},
            {"id":"pilot_test","label":"Open to pilot-testing new intermodal solutions?","type":"radio","options":["Yes","No"]},
            {"id":"govt_influence","label":"Influence of government campaigns (1–5)","type":"likert5"},
            {"id":"policy_suggestions","label":"Policy suggestions (open text)","type":"textarea"},
        ],
    },
    {
        "section": "Section 7: Stated-Preference Choice Tasks",
        "items": [
            # Context vignette card will be injected in renderer for this section
            {
                "id": "sp_task_1",
                "label": "Task 1: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 90, "Departures per day": 6},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 600, "Transit time (h)": 18, "On-time reliability (% within ±2h)": 95, "Departures per day": 4},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_2",
                "label": "Task 2: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 300, "Transit time (h)": 8,  "On-time reliability (% within ±2h)": 85, "Departures per day": 4},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 95, "Departures per day": 6},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_3",
                "label": "Task 3: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 750, "Transit time (h)": 24, "On-time reliability (% within ±2h)": 85, "Departures per day": 1},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 600, "Transit time (h)": 18, "On-time reliability (% within ±2h)": 95, "Departures per day": 4},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_4",
                "label": "Task 4: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 600, "Transit time (h)": 18, "On-time reliability (% within ±2h)": 95, "Departures per day": 4},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 600, "Transit time (h)": 24, "On-time reliability (% within ±2h)": 98, "Departures per day": 2},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_5",
                "label": "Task 5: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 600, "Transit time (h)": 24, "On-time reliability (% within ±2h)": 98, "Departures per day": 2},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 90, "Departures per day": 6},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_6",
                "label": "Task 6: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 90, "Departures per day": 6},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 300, "Transit time (h)": 8,  "On-time reliability (% within ±2h)": 85, "Departures per day": 2},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_7",
                "label": "Task 7: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 300, "Transit time (h)": 8,  "On-time reliability (% within ±2h)": 85, "Departures per day": 4},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 600, "Transit time (h)": 24, "On-time reliability (% within ±2h)": 98, "Departures per day": 1},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            {
                "id": "sp_task_8",
                "label": "Task 8: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 90, "Departures per day": 6},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 300, "Transit time (h)": 8,  "On-time reliability (% within ±2h)": 85, "Departures per day": 4},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            # Dominance check (Intermodal strictly better on all 4 attributes)
            {
                "id": "sp_task_9_dominance",
                "label": "Check A: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 600, "Transit time (h)": 18, "On-time reliability (% within ±2h)": 90, "Departures per day": 2},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 95, "Departures per day": 4},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
            # Consistency check (repeat Task 2)
            {
                "id": "sp_task_10_consistency",
                "label": "Check B: Please choose your preferred option",
                "type": "choice_table",
                "alternatives": [
                    {"alt": "Road",       "Cost (CHF/TEU)": 300, "Transit time (h)": 8,  "On-time reliability (% within ±2h)": 85, "Departures per day": 4},
                    {"alt": "Intermodal", "Cost (CHF/TEU)": 450, "Transit time (h)": 12, "On-time reliability (% within ±2h)": 95, "Departures per day": 6},
                    {"alt": "Status-quo / Opt-out", "Cost (CHF/TEU)": "—", "Transit time (h)": "—", "On-time reliability (% within ±2h)": "—", "Departures per day": "—"},
                ],
            },
        ],
    }
]


def card(title=None, body_html=""):
    st.markdown(
        f"""
        <div class="card">
            {f"<h3 style='margin:0 0 8px 0;'>{title}</h3>" if title else ""}
            <div>{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_question_label(text):
    st.markdown(f'<div class="q-label">{text}</div>', unsafe_allow_html=True)

def display_multiple_choice(question, options, key):
    st.markdown(f'<div class="qa-card"><div class="qa-title">{question}</div>', unsafe_allow_html=True)
    val = st.radio(label="", options=options, key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def likert5(key, label):
    st.markdown(f'<div class="qa-card"><div class="qa-title">{label}</div>', unsafe_allow_html=True)
    val = st.slider("", 1, 5, 3, key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def matrix_likert5(prefix, rows):
    display_question_label("Please rate each item (1–5)")
    vals = {}
    for r in rows:
        st.markdown(f'<div class="qa-card"><div class="qa-title">{r}</div>', unsafe_allow_html=True)
        vals[r] = st.slider("", 1, 5, 3, key=f"{prefix}:{r}", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    return vals

def matrix_ordinal(prefix, label, rows, cols):
    st.markdown(f'<div class="matrix-label"><b>{label}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-hint"><i>Please select one option per row</i></div>', unsafe_allow_html=True)

    vals = {}
    for r in rows:
        st.markdown(f'<div class="matrix-row"><b>{r}</b></div>', unsafe_allow_html=True)
        vals[r] = st.radio("", cols, horizontal=True, key=f"{prefix}:{r}", label_visibility="collapsed")
    return vals

def display_multiselect(question, options, key):
    st.markdown(f'<div class="qa-card"><div class="qa-title">{question}</div>', unsafe_allow_html=True)
    val = st.multiselect("", options, key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def display_text(question, key):
    st.markdown(f'<div class="qa-card"><div class="qa-title">{question}</div>', unsafe_allow_html=True)
    val = st.text_input("", key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def display_textarea(question, key):
    st.markdown(f'<div class="qa-card"><div class="qa-title">{question}</div>', unsafe_allow_html=True)
    val = st.text_area("", key=key, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def display_choice_table(question, alternatives, key):
    """
    Render a stated-preference choice task as a table + radio buttons.
    Shows only Road & Intermodal rows. Adds an "Other" option to the selector
    without a table row (no attribute data shown for Other).
    Also auto-renames any legacy 'Status-quo / Opt-out' to 'Other'.
    """
    # Outer card
    st.markdown(f'<div class="qa-card"><div class="qa-title">{question}</div>', unsafe_allow_html=True)

    # Force white background for the table
    st.markdown(
        """
        <style>
        .qa-card .stTable, .qa-card .stDataFrame {
            background-color: white !important;
            border-radius: 8px;
            padding: 6px;
            margin-bottom: 8px;
        }
        .qa-card .stTable thead tr th { background-color: white !important; color: black !important; }
        .qa-card .stTable tbody tr td { background-color: white !important; color: black !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Filter out any Opt-out rows from the table and keep only Road/Intermodal
    filtered_alts = []
    for a in alternatives:
        alt_name = a.get("alt", "")
        if alt_name.lower() in ["status-quo / opt-out", "status-quo", "opt-out", "other"]:
            continue
        filtered_alts.append(a)

    df = pd.DataFrame(filtered_alts)
    cols = ["alt"] + [c for c in SP_COLUMNS if c in df.columns]
    if not df.empty:
        df = df[[c for c in cols if c in df.columns]]
        st.table(df)

    # Radio options include visible alternatives + 'Other'
    radio_options = [alt.get("alt", "") for alt in filtered_alts]
    if "Other" not in radio_options:
        radio_options.append("Other")

    val = st.radio(
        "Select your preferred option:",
        options=radio_options,
        key=key,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    return val


def answers_to_row(answers: dict, schema: list) -> dict:
    """
    Convert nested answers to a flat row where keys = question labels and values = user selections.
    (Note: This remains compact. For model-ready long format, export should reshape later.)
    """
    row = {}

    for section in schema:
        for item in section["items"]:
            qid   = item["id"]
            label = item["label"]
            typ   = item["type"]
            val   = answers.get(qid, None)

            if typ in ("text", "textarea", "radio"):
                row[label] = val if val is not None else ""

            elif typ == "multiselect":
                row[label] = ", ".join(val) if val else ""

            elif typ == "likert5":
                row[label] = int(val) if val is not None else None

            elif typ == "matrix_likert5":
                for r_label, r_val in (val or {}).items():
                    row[f"{label} | {r_label}"] = int(r_val) if r_val is not None else None

            elif typ == "matrix_ordinal":
                for r_label, r_val in (val or {}).items():
                    row[f"{label} | {r_label}"] = r_val if r_val is not None else ""

            elif typ == "choice_table":
                row[label] = val if val is not None else ""
            else:
                row[label] = val if val is not None else ""

    return row


def save_to_firebase(row: dict):
    try:
        client = init_firestore()
        doc_ref = client.collection("ch_intermodal_survey_rows").document()
        doc_ref.set(row)
        return True, doc_ref.id
    except Exception as e:
        return False, str(e)


# Background image
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Header
def display_header():
    st.markdown(
        """
        <div style="text-align: center; padding: 2px;">
            <img src="https://logowik.com/content/uploads/images/eth-zurich1144.jpg" width="100">
            <h1 style="color: white;">Survey: Swiss Intermodal Freight</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
def display_footer():
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; background-color: rgba(255, 255, 255); border-radius: 10px; margin-top: 20px;">
            <p style="color: black;">© 2025 ETH Zurich. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    set_background("project-mobility.jpg")
    display_header()

    # CSS tune-up
    st.markdown(
        """
        <style>
        .stRadio > div { background: rgba(255,255,255,0.92); padding: 8px; border-radius: 6px; }
        div[data-testid=\"stSlider\"] { background: rgba(255,255,255,0.92); padding: 6px; border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Intro card
    if st.session_state.get("page_idx", 0) != -1:
        card(
            "About this survey",
            "This pilot study explores short-distance intermodal rail adoption in Switzerland. "
            "Your responses will inform a discrete choice model and an optimization study. "
            "It takes ~8–10 minutes. Thank you!"
        )

    # Init session state
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}

    # Success page branch
    if st.session_state.page_idx == -1:
        st.markdown(
            """
            <div style=\"padding: 20px; background-color: rgba(255, 255, 255, 0.9); 
                        border-radius: 10px; margin: 50px auto; max-width: 600px; text-align: center;\">
                <h2 style=\"color: green;\">Thank You!</h2>
                <p style=\"font-size: 18px;color: green;\">Your survey has been submitted successfully.</p>
                <p style=\"color: green;\">Your responses will help improve our research.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # --- Define pages BEFORE sidebar to avoid undefined refs on rerun ---
    pages = survey_schema
    section_names = [p["section"] for p in pages]
    total_pages = len(pages)
    current_idx = st.session_state.page_idx

    # Sidebar jump-to controls
    # Ensure a unique key namespace for nav widgets per session
    if 'nav_uid' not in st.session_state:
        st.session_state.nav_uid = str(uuid.uuid4())

    with st.sidebar:
        st.markdown("### Navigate")
        st.caption(f"Page {current_idx + 1} of {total_pages}")
        jump_key = f"jump_to_section_select_{st.session_state.nav_uid}"
        jump_target = st.selectbox(
            "Jump to section",
            section_names,
            index=current_idx,
            key=jump_key,
        )
        if jump_target != section_names[current_idx]:
            st.session_state.page_idx = section_names.index(jump_target)
            st.rerun()
        if st.button("Go to SP Tasks", key=f"go_sp_tasks_{st.session_state.nav_uid}"):
            sp_idx = section_names.index("Section 7: Stated-Preference Choice Tasks")
            st.session_state.page_idx = sp_idx
            st.rerun()

    # Current section (defined AFTER sidebar to use possibly updated page_idx)
    section = pages[st.session_state.page_idx]

    # Inject vignette before SP tasks
    if section["section"] == "Section 7: Stated-Preference Choice Tasks":
        card(
            title="Reference shipment (for the following tasks)",
            body_html=(
                "Domestic move, <b>OD 120–180 km</b>, <b>1× 20’ container</b> (general cargo), "
                "<b>medium urgency</b> (delivery within 24h acceptable). Compare options below. "
                "<br><i>On-time reliability = % delivered within ±2 hours of promised time.</i>"
            ),
        )

    card(title=f"{section['section']} — Page {st.session_state.page_idx+1} / {total_pages}")

    # Collect inputs
    if section["section"] != "Section 7: Stated-Preference Choice Tasks":
        for item in section["items"]:
            t = item["type"]
            key = item["id"]
            label = item["label"]
            if t == "radio":
                st.session_state.answers[key] = display_multiple_choice(label, item["options"], key)
            elif t == "multiselect":
                st.session_state.answers[key] = display_multiselect(label, item["options"], key)
            elif t == "text":
                st.session_state.answers[key] = display_text(label, key)
            elif t == "textarea":
                st.session_state.answers[key] = display_textarea(label, key)
            elif t == "likert5":
                st.session_state.answers[key] = likert5(key, label)
            elif t == "matrix_likert5":
                st.session_state.answers[key] = matrix_likert5(key, item["rows"])
            elif t == "matrix_ordinal":
                st.session_state.answers[key] = matrix_ordinal(key, label, item["rows"], item["cols"])
            elif t == "choice_table":
                st.session_state.answers[key] = display_choice_table(label, item["alternatives"], key)
            else:
                st.info(f"Unknown field type: {t}")
    else:
        # Dynamic SP rendering from CSV designs
        ensure_respondent_context()
        core_df, checks_df = load_sp_designs()
        if core_df is None:
            st.error("SP design files not found. Please place sp_core_design_blocks.csv and sp_checks_design.csv in /mnt/data or app working directory.")
        else:
            blk = st.session_state.assigned_block
            st.info(f"You have been assigned to Block {blk} (8 tasks).")
            tasks = build_tasks_for_block(core_df, checks_df, blk)
            for tnum, alts, is_check, ctype in tasks:
                label = f"Task {tnum}: Please choose your preferred option" if not is_check else f"Check ({ctype.title()}): Please choose your preferred option"
                key = f"sp_choice_{tnum}"
                st.session_state.answers[key] = display_choice_table(label, alts, key)

    # Navigation buttons
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        if st.session_state.page_idx > 0 and st.button("Previous"):
            st.session_state.page_idx -= 1
            st.rerun()

    with nav3:
        is_last = st.session_state.page_idx == len(pages) - 1
        if not is_last:
            if st.button("Next"):
                st.session_state.page_idx += 1
                st.rerun()
        else:
            if st.button("Submit"):
                flat_row = answers_to_row(st.session_state.answers, survey_schema)
                flat_row["submitted_at_utc"] = datetime.utcnow().isoformat() + "Z"
                flat_row["schema_version"] = "v2"
                flat_row["app_version"] = "2025-10-10"
                flat_row["respondent_id"] = st.session_state.get("respondent_id")
                flat_row["assigned_block"] = st.session_state.get("assigned_block")
                flat_row["started_at_utc"] = st.session_state.get("started_at_utc")

                # Build long-format SP rows
                long_rows = []
                core_df, checks_df = load_sp_designs()
                if core_df is not None:
                    blk = st.session_state.get("assigned_block", 1)
                    tasks = build_tasks_for_block(core_df, checks_df, blk)
                    for tnum, alts, is_check, ctype in tasks:
                        choice = st.session_state.answers.get(f"sp_choice_{tnum}")
                        timestamp = datetime.utcnow().isoformat() + "Z"
                        # Two alternatives with attributes
                        for alt in alts:
                            rec = {
                                "respondent_id": flat_row["respondent_id"],
                                "block_id": blk,
                                "task_num": int(tnum),
                                "alt_id": alt["alt"],
                                "cost_chf_teu": int(alt["Cost (CHF/TEU)"]),
                                "time_h": int(alt["Transit time (h)"]),
                                "ontime_p": int(alt["On-time reliability (% within ±2h)"]),
                                "dep_per_day": int(alt["Departures per day"]),
                                "is_check": 1 if is_check else 0,
                                "check_type": ctype,
                                "chosen": 1 if choice == alt["alt"] else 0,
                                "answered_at_utc": timestamp,
                            }
                            long_rows.append(rec)
                        # Add explicit 'Other' row without attributes
                        long_rows.append({
                            "respondent_id": flat_row["respondent_id"],
                            "block_id": blk,
                            "task_num": int(tnum),
                            "alt_id": "Other",
                            "cost_chf_teu": None,
                            "time_h": None,
                            "ontime_p": None,
                            "dep_per_day": None,
                            "is_check": 1 if is_check else 0,
                            "check_type": ctype,
                            "chosen": 1 if choice == "Other" else 0,
                            "answered_at_utc": timestamp,
                        })

                ok_long, info_long = save_long_rows_to_firebase(long_rows) if long_rows else (True, "none")
                ok_flat, info_flat = save_to_firebase(flat_row)

                if ok_flat and ok_long:
                    st.success("Thank you! Your response has been recorded.")
                    st.session_state.answers = {}
                    st.session_state.page_idx = -1
                else:
                    st.error(f"Could not save responses: flat={info_flat}; long={info_long}")

    display_footer()


if __name__ == "__main__":
    main()
