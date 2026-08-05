/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  ExternalLink,
  HardDrive,
  HelpCircle,
  Layers,
  Monitor,
  ShoppingCart,
  Wrench,
} from 'lucide-react';
import type {
  BasketSnapshot,
  Deal,
  DealsSnapshot,
  ProfileFit,
  ProfileId,
  ProfileInfo,
} from './types';

/**
 * Static reader of the published snapshot. No API calls, no credentials — the
 * pipeline runs locally and commits `deals.json`, this just renders it.
 */
const SNAPSHOT_URL = `${import.meta.env.BASE_URL}deals.json`;

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; snapshot: DealsSnapshot };

/**
 * Budget presets. Applied here, at presentation, and never at ingest.
 *
 * Filtering the scrape to a budget would bias the market index: a median
 * computed only over cheap listings sits low, so everything would look
 * overpriced against it and the filter would corrupt the number used to judge.
 * Affordability is a question asked at the end.
 */
const BUDGETS = [
  { label: 'Any price', value: null },
  { label: '≤ 6 000 kr', value: 6000 },
  { label: '≤ 8 000 kr', value: 8000 },
  { label: '≤ 10 000 kr', value: 10000 },
];

export default function App() {
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const [category, setCategory] = useState<string>('All');
  const [profileId, setProfileId] = useState<ProfileId>('ai-workstation');
  const [budget, setBudget] = useState<number | null>(null);
  const [showUnfit, setShowUnfit] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(SNAPSHOT_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json() as Promise<DealsSnapshot>;
      })
      .then((snapshot) => {
        if (!cancelled) setState({ status: 'ready', snapshot });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const snapshot = state.status === 'ready' ? state.snapshot : null;

  const categories = useMemo(() => {
    if (!snapshot) return ['All'];
    return ['All', ...Array.from(new Set(snapshot.deals.map((d) => d.category))).sort()];
  }, [snapshot]);

  const profile = useMemo(
    () => snapshot?.profiles.find((p) => p.id === profileId) ?? null,
    [snapshot, profileId],
  );

  const { visible, hiddenByProfile } = useMemo(() => {
    if (!snapshot) return { visible: [] as Deal[], hiddenByProfile: 0 };

    let deals = snapshot.deals;
    if (category !== 'All') deals = deals.filter((d) => d.category === category);
    if (budget !== null) {
      // Compared against the *effective* price: a 6 400 kr machine that needs
      // 800 kr of RAM does not fit a 6 000 kr budget, and showing it as if it
      // did is the one filtering mistake that actually costs money.
      deals = deals.filter((d) => effectivePrice(d, profileId) <= budget);
    }

    const unfit = deals.filter((d) => d.fits[profileId]?.verdict === 'unfit');
    if (!showUnfit) deals = deals.filter((d) => d.fits[profileId]?.verdict !== 'unfit');

    return {
      visible: [...deals].sort((a, b) => compare(a, b, profileId)),
      hiddenByProfile: showUnfit ? 0 : unfit.length,
    };
  }, [snapshot, category, profileId, budget, showUnfit]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-emerald-500/30">
      <Navbar generatedAt={snapshot?.generatedAt ?? null} />

      <main className="max-w-7xl mx-auto px-4 py-8 relative">
        <div className="absolute inset-0 bg-grid-pattern opacity-30 select-none pointer-events-none -z-10 [mask-image:linear-gradient(to_bottom,white,transparent)]" />

        {state.status === 'loading' && <Notice>Loading deals…</Notice>}

        {state.status === 'error' && (
          <Notice tone="warn">
            <p className="font-medium text-zinc-200">No snapshot found.</p>
            <p className="mt-1">
              Generate one by running the local pipeline, which writes{' '}
              <code className="text-emerald-400">web/public/deals.json</code>.
            </p>
            <p className="mt-2 text-xs text-zinc-600 font-mono">{state.message}</p>
          </Notice>
        )}

        {snapshot && (
          <>
            <ProfileSelector
              profiles={snapshot.profiles}
              selected={profileId}
              onSelect={setProfileId}
            />

            <KpiRow deals={snapshot.deals} />

            {snapshot.baskets?.[profileId] && (
              <BuildOrBuyPanel basket={snapshot.baskets[profileId]} />
            )}

            <div className="flex items-center gap-2 mb-6 border-b border-zinc-800 pb-4 flex-wrap">
              {categories.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setCategory(tab)}
                  className={`px-4 py-1.5 rounded-md text-sm transition-all duration-200 ${
                    category === tab
                      ? 'bg-zinc-800 text-zinc-50 font-medium shadow-sm border border-zinc-700'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 border border-transparent'
                  }`}
                >
                  {tab}
                </button>
              ))}

              <div className="ml-auto flex items-center gap-2">
                <select
                  value={budget ?? ''}
                  onChange={(e) => setBudget(e.target.value ? Number(e.target.value) : null)}
                  className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-1.5 text-sm text-zinc-300"
                >
                  {BUDGETS.map((b) => (
                    <option key={b.label} value={b.value ?? ''}>
                      {b.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {visible.length === 0 ? (
              <Notice>No listings match this profile and budget yet.</Notice>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {visible.map((deal) => (
                  <DealCard key={deal.id} deal={deal} fit={deal.fits[profileId] ?? null} />
                ))}
              </div>
            )}

            {hiddenByProfile > 0 && (
              <button
                onClick={() => setShowUnfit(true)}
                className="mt-4 text-sm text-zinc-500 hover:text-zinc-300 underline"
              >
                {hiddenByProfile} listing{hiddenByProfile === 1 ? '' : 's'} hidden — they
                fail {profile?.label ?? 'this profile'}&apos;s hardware requirements. Show
                them anyway.
              </button>
            )}

            <Footer snapshot={snapshot} shown={visible.length} />
          </>
        )}
      </main>
    </div>
  );
}

const VERDICT_TIER: Record<string, number> = {
  fits: 0,
  'fits-after-upgrade': 1,
  unverified: 1,
  unfit: 2,
};

function effectivePrice(deal: Deal, profileId: ProfileId): number {
  return deal.fits[profileId]?.effectivePriceSek ?? deal.priceSek;
}

/**
 * Ordering: profile fit, then the profile's own metric, then market value.
 *
 * Deliberately a sequence of independent comparisons rather than one blended
 * score. The opaque 0–100 `deal_score` was removed from this project because
 * nobody could say why one listing outranked another, and a weighted sum here
 * would quietly bring it back.
 *
 * It would also be arithmetically wrong. `deltaSek` is negative for a good
 * deal, so multiplying it by a fitness factor below 1 correctly penalises
 * bargains and simultaneously *rewards* overpriced listings, because scaling a
 * positive delta toward zero moves it up the board. One operator, two opposite
 * meanings. Mirrors `rank_key` in `pipeline/profiles.py`.
 */
function compare(a: Deal, b: Deal, profileId: ProfileId): number {
  const fa = a.fits[profileId];
  const fb = b.fits[profileId];

  const tier = (VERDICT_TIER[fa?.verdict ?? 'unfit'] ?? 2) - (VERDICT_TIER[fb?.verdict ?? 'unfit'] ?? 2);
  if (tier !== 0) return tier;

  const ma = fa?.metricValue ?? Number.POSITIVE_INFINITY;
  const mb = fb?.metricValue ?? Number.POSITIVE_INFINITY;
  if (ma !== mb) return ma - mb;

  return value(a) - value(b);
}

/**
 * The deterministic price delta, scaled by the model's confidence.
 *
 * The AI adjusts how much to trust a number; it never produces one. Deals we
 * couldn't price at all sort last rather than pretending to be neutral.
 */
function value(deal: Deal): number {
  if (deal.math.deltaSek === null) return Number.POSITIVE_INFINITY;
  return deal.math.deltaSek * (deal.ai?.confidence ?? 1);
}

function Navbar({ generatedAt }: { generatedAt: string | null }) {
  return (
    <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md border border-zinc-700 flex items-center justify-center bg-zinc-900 shadow-[0_0_15px_rgba(16,185,129,0.15)] ring-1 ring-emerald-500/20">
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h1 className="font-semibold text-lg tracking-tight">Sentinel-Blocket</h1>
            <p className="text-[10px] text-zinc-500 font-mono tracking-wider uppercase">
              Used PC deal index · Sweden
            </p>
          </div>
        </div>
        {generatedAt && (
          <p className="text-xs text-zinc-500 font-mono">
            updated {new Date(generatedAt).toLocaleString('sv-SE')}
          </p>
        )}
      </div>
    </nav>
  );
}

/**
 * The profile selector, with each profile's reasoning shown rather than hidden.
 *
 * A filter that silently drops listings the user can see on Blocket is
 * infuriating and looks broken. One that says "8 GB VRAM, below this profile's
 * floor, and here is why that floor exists" is the actual product.
 */
function ProfileSelector({
  profiles,
  selected,
  onSelect,
}: {
  profiles: ProfileInfo[];
  selected: ProfileId;
  onSelect: (id: ProfileId) => void;
}) {
  const active = profiles.find((p) => p.id === selected);

  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 flex-wrap mb-3">
        <span className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mr-1">
          Building for
        </span>
        {profiles.map((p) => (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            className={`px-4 py-1.5 rounded-full text-sm transition-all duration-200 border ${
              selected === p.id
                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 font-medium'
                : 'text-zinc-400 border-zinc-800 hover:text-zinc-200 hover:border-zinc-700'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {active && (
        <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-4">
          <p className="text-sm text-zinc-400 leading-relaxed">{active.rationale}</p>
          <div className="flex flex-wrap gap-2 mt-3">
            {active.requirements.map((r) => (
              <span
                key={r.label}
                title={r.why}
                className="text-[11px] px-2 py-0.5 rounded-full border bg-zinc-800/60 text-zinc-300 border-zinc-700 font-mono"
              >
                {r.label}
                {r.upgradeable && <span className="text-zinc-500 ml-1">(upgradeable)</span>}
              </span>
            ))}
            {active.metric && (
              <span className="text-[11px] px-2 py-0.5 rounded-full border bg-emerald-500/5 text-emerald-400/80 border-emerald-500/20 font-mono">
                ranked by {active.metric} ({active.metricUnit})
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Build-versus-buy, side by side.
 *
 * Neither option is recommended. The point is to make the tradeoff concrete —
 * a saving spread across four strangers in four towns is a different
 * proposition from the same saving in one transaction, and only the buyer can
 * price that difference.
 */
function BuildOrBuyPanel({ basket }: { basket: BasketSnapshot }) {
  const comparable = basket.savingSek !== null;

  return (
    <div className="bg-zinc-900/40 border border-zinc-800/80 rounded-xl p-5 mb-8">
      <div className="flex items-center gap-2 mb-4">
        <ShoppingCart className="w-4 h-4 text-zinc-500" />
        <h3 className="text-sm font-medium text-zinc-300">Buy assembled, or build it?</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        <Field label="Cheapest qualifying prebuilt">
          <span className="text-2xl font-mono text-zinc-100">
            {basket.bestPrebuiltPriceSek === null ? (
              <span className="text-sm text-zinc-500">none listed right now</span>
            ) : (
              <>
                {basket.bestPrebuiltPriceSek.toLocaleString('sv-SE')}
                <span className="text-sm text-zinc-500 ml-1">kr</span>
              </>
            )}
          </span>
        </Field>

        <Field label={`Parts basket · ${basket.sellerCount} sellers`}>
          <span className="text-2xl font-mono text-zinc-100">
            {basket.totalSek.toLocaleString('sv-SE')}
            <span className="text-sm text-zinc-500 ml-1">kr</span>
          </span>
        </Field>

        <Field label="Difference">
          {comparable ? (
            <span
              className={`text-2xl font-mono ${
                (basket.savingSek ?? 0) > 0 ? 'text-emerald-400' : 'text-zinc-400'
              }`}
            >
              {Math.abs(basket.savingSek ?? 0).toLocaleString('sv-SE')}
              <span className="text-sm text-zinc-500 ml-1">
                kr {(basket.savingSek ?? 0) > 0 ? 'cheaper to build' : 'cheaper assembled'}
              </span>
            </span>
          ) : (
            <span className="text-sm text-zinc-500">
              not comparable — no {basket.missingSlots.join(', ').toLowerCase()} listed
            </span>
          )}
        </Field>
      </div>

      {basket.items.length > 0 && (
        <div className="mt-4 pt-4 border-t border-zinc-800/80 space-y-1">
          {basket.items.map((item) => (
            <div key={item.url} className="flex items-center justify-between text-sm font-mono">
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer nofollow"
                className="text-zinc-400 hover:text-emerald-400 truncate"
              >
                <span className="text-zinc-600 mr-2">{item.slot}</span>
                {item.name}
                {item.city && <span className="text-zinc-600 ml-2">· {item.city}</span>}
              </a>
              <span className="text-zinc-300 shrink-0 ml-2">
                {item.priceSek.toLocaleString('sv-SE')} kr
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Not a footnote. A basket total that omits three real costs is cheaper
          by exactly the amount it omitted, and that tilts every comparison
          toward building unless it is said out loud. */}
      <p className="mt-3 text-xs text-zinc-600">
        Basket excludes {basket.excludedSlots.join(', ').toLowerCase()} — these are
        rarely listed separately in a form the extractor can read, so a built machine
        will cost more than the figure above.
      </p>
    </div>
  );
}

function KpiRow({ deals }: { deals: Deal[] }) {
  const priced = deals.filter((d) => d.math.deltaSek !== null);
  const underMarket = priced.filter((d) => (d.math.deltaSek ?? 0) < 0);
  const totalUpside = underMarket.reduce((sum, d) => sum + Math.abs(d.math.deltaSek ?? 0), 0);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      <Kpi label="Deals tracked" icon={<Layers className="w-4 h-4 text-zinc-500" />}>
        <span className="text-zinc-100">{deals.length}</span>
        <p className="text-xs text-zinc-500 mt-2 font-mono">
          {priced.length} with enough data to price
        </p>
      </Kpi>

      <Kpi label="Below market" icon={<Activity className="w-4 h-4 text-emerald-400" />}>
        <span className="text-emerald-400">{underMarket.length}</span>
        <p className="text-xs text-zinc-500 mt-2 font-mono">vs. component medians</p>
      </Kpi>

      <Kpi label="Total upside" icon={<HardDrive className="w-4 h-4 text-blue-400" />}>
        <span className="text-blue-400">
          {totalUpside.toLocaleString('sv-SE')}
          <span className="text-lg text-zinc-500 ml-1">kr</span>
        </span>
        <p className="text-xs text-zinc-500 mt-2">Summed across below-market listings</p>
      </Kpi>
    </div>
  );
}

function Kpi({
  label,
  icon,
  children,
}: {
  label: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-xl p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-zinc-400 text-sm font-medium">{label}</h3>
        {icon}
      </div>
      <div className="text-3xl font-mono tracking-tight">{children}</div>
    </div>
  );
}

function DealCard({ deal, fit }: { deal: Deal; fit: ProfileFit | null }) {
  const { math, ai } = deal;
  const delta = math.deltaSek;

  return (
    <div className="group flex flex-col bg-[#111113] border border-zinc-800/80 hover:border-zinc-700 rounded-xl p-5 transition-all duration-300">
      <div className="flex justify-between items-start mb-4 gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 font-medium">
              {deal.category}
            </span>
            {deal.city && <span className="text-xs text-zinc-500 font-mono">{deal.city}</span>}
          </div>
          <h2 className="text-lg font-medium text-zinc-100 group-hover:text-emerald-400 transition-colors line-clamp-2">
            {deal.title}
          </h2>
        </div>
        <DeltaBadge delta={delta} sampleCount={math.minSampleCount} />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <Field label="Asking price">
          <span className="text-xl font-mono text-zinc-50">
            {deal.priceSek.toLocaleString('sv-SE')}
            <span className="text-sm text-zinc-500 ml-1">kr</span>
          </span>
        </Field>
        <Field label="Parts value (median)">
          <span className="text-xl font-mono text-zinc-300">
            {math.estimatedPartsSek === null ? (
              <span className="text-sm text-zinc-500">insufficient data</span>
            ) : (
              <>
                {math.estimatedPartsSek.toLocaleString('sv-SE')}
                <span className="text-sm text-zinc-500 ml-1">kr</span>
              </>
            )}
          </span>
        </Field>
      </div>

      {fit && <FitPanel fit={fit} />}

      {/*
        Every component, never a slice. This showed the first four of up to
        nine while the parts total beside it was summed over all of them, so
        the card could not be checked by eye — and it was hiding real errors:
        one listing's total was mostly a 5 500 kr "1tb ssd" plus a duplicate of
        that same drive, neither of them on screen. For a site whose whole
        claim is "here is the arithmetic", an unverifiable parts list is the
        wrong default.

        Unpriced parts are dimmed rather than dropped: "no market data for this
        one" is itself the explanation for a total that looks low.
      */}
      {math.components.length > 0 && (
        <div className="mb-4 space-y-1">
          {math.components.map((c, i) => (
            <div
              key={i}
              className={`flex items-center justify-between text-sm font-mono ${
                c.medianSek === null ? 'opacity-50' : ''
              }`}
            >
              <span className="flex items-center gap-1.5 text-zinc-300 truncate">
                {c.type === 'GPU' ? (
                  <Monitor className="w-3 h-3 text-zinc-500 shrink-0" />
                ) : (
                  <Cpu className="w-3 h-3 text-zinc-500 shrink-0" />
                )}
                {c.name}
              </span>
              <span className="text-zinc-500 shrink-0 ml-2">
                {c.medianSek === null ? (
                  <span title="No market data for this part — it adds nothing to the total">
                    no data
                  </span>
                ) : (
                  <>
                    {c.medianSek.toLocaleString('sv-SE')} kr
                    <span
                      className={
                        c.sampleCount < 5
                          ? 'text-amber-600/80 ml-1'
                          : 'text-zinc-700 ml-1'
                      }
                      title={
                        c.sampleCount < 5
                          ? `Only ${c.sampleCount} observation(s) — provisional`
                          : `${c.sampleCount} observations`
                      }
                    >
                      n={c.sampleCount}
                    </span>
                  </>
                )}
              </span>
            </div>
          ))}
        </div>
      )}

      {ai && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 mb-4">
          <p className="text-sm text-zinc-400 leading-relaxed">
            <span className="text-emerald-400 font-medium text-xs tracking-wider uppercase mr-2">
              Assessment
            </span>
            {ai.summary}
          </p>
          {ai.flags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {ai.flags.map((f) => (
                <span
                  key={f.id}
                  className={`text-[11px] px-2 py-0.5 rounded-full border font-medium ${
                    f.severity === 'high'
                      ? 'bg-red-500/15 text-red-400 border-red-500/30 uppercase tracking-wide'
                      : f.severity === 'warn'
                        ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                        : 'bg-zinc-800 text-zinc-400 border-zinc-700'
                  }`}
                >
                  {f.severity !== 'info' && (
                    <AlertTriangle className="w-3 h-3 inline mr-1 -mt-0.5" />
                  )}
                  {f.label}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between mt-auto pt-4 border-t border-zinc-800/80">
        <span className="text-xs font-mono text-zinc-600">
          {deal.condition ?? 'condition not stated'}
        </span>
        <a
          href={deal.url}
          target="_blank"
          rel="noopener noreferrer nofollow"
          className="px-3 py-1.5 text-sm bg-zinc-100 text-zinc-900 hover:bg-white rounded-md font-medium transition-colors flex items-center gap-1.5"
        >
          View listing <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
}

/**
 * How the listing measures up against the selected profile.
 *
 * Renders the three states separately on purpose. "We could not tell" is shown
 * as its own thing, in its own colour, because most ads state the graphics card
 * and nothing else — and presenting that silence as a failure would be the same
 * error as counting an unpriced component as free.
 */
function FitPanel({ fit }: { fit: ProfileFit }) {
  const tone =
    fit.verdict === 'fits'
      ? 'border-emerald-500/20 bg-emerald-500/5'
      : fit.verdict === 'unfit'
        ? 'border-red-500/20 bg-red-500/5'
        : 'border-zinc-800 bg-zinc-900';

  return (
    <div className={`rounded-lg border p-3 mb-4 ${tone}`}>
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex flex-wrap gap-1.5">
          {fit.met.map((label) => (
            <Chip key={label} tone="ok" icon={<CheckCircle2 className="w-3 h-3" />}>
              {label}
            </Chip>
          ))}
          {fit.unmet.map((label) => (
            <Chip
              key={label}
              tone={fit.blockedBy.includes(label) ? 'bad' : 'warn'}
              icon={
                fit.blockedBy.includes(label) ? (
                  <AlertTriangle className="w-3 h-3" />
                ) : (
                  <Wrench className="w-3 h-3" />
                )
              }
            >
              {label}
            </Chip>
          ))}
          {fit.unknown.map((label) => (
            <Chip key={label} tone="unknown" icon={<HelpCircle className="w-3 h-3" />}>
              {label}?
            </Chip>
          ))}
        </div>

        {fit.metricValue !== null && (
          <span className="text-sm font-mono text-zinc-300 shrink-0">
            {Math.round(fit.metricValue).toLocaleString('sv-SE')}
            <span className="text-zinc-500 ml-1 text-xs">{fit.metricUnit}</span>
          </span>
        )}
      </div>

      {fit.upgradeCostSek > 0 && (
        <p className="text-xs text-zinc-400 mt-2">
          + {fit.upgradeCostSek.toLocaleString('sv-SE')} kr to bring it up to spec —
          effectively{' '}
          <span className="text-zinc-200 font-mono">
            {fit.effectivePriceSek.toLocaleString('sv-SE')} kr
          </span>
        </p>
      )}

      {fit.vramCaveat && (
        <p className="text-xs text-amber-500/70 mt-2">{fit.vramCaveat}</p>
      )}
    </div>
  );
}

function Chip({
  children,
  tone,
  icon,
}: {
  children: ReactNode;
  tone: 'ok' | 'warn' | 'bad' | 'unknown';
  icon: ReactNode;
}) {
  const styles = {
    ok: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    warn: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    bad: 'bg-red-500/10 text-red-400 border-red-500/20',
    unknown: 'bg-zinc-800 text-zinc-500 border-zinc-700',
  }[tone];

  return (
    <span
      className={`text-[11px] px-2 py-0.5 rounded-full border font-medium inline-flex items-center gap-1 ${styles}`}
    >
      {icon}
      {children}
    </span>
  );
}

/**
 * The headline number. Shows sample count alongside, because a large delta
 * backed by n=2 deserves less trust than a small one backed by n=40 — hiding
 * that would make the figure look more certain than it is.
 */
function DeltaBadge({ delta, sampleCount }: { delta: number | null; sampleCount: number }) {
  if (delta === null) {
    return (
      <div className="px-3 py-2 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-500 text-xs text-center shrink-0">
        no data
      </div>
    );
  }

  const below = delta < 0;
  const lowConfidence = sampleCount < 5;

  return (
    <div
      className={`px-3 py-2 rounded-lg border font-mono flex flex-col items-center justify-center min-w-[96px] shrink-0 ${
        below
          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
          : 'bg-zinc-800/50 text-zinc-400 border-zinc-700'
      }`}
    >
      <span className="text-[10px] opacity-80 leading-none mb-1">
        {below ? 'BELOW MARKET' : 'ABOVE MARKET'}
      </span>
      <span className="text-lg font-bold leading-none">
        {Math.abs(delta).toLocaleString('sv-SE')} kr
      </span>
      {lowConfidence && (
        <span className="text-[10px] text-amber-500/80 mt-1 leading-none">low confidence</span>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1 font-semibold">{label}</p>
      {children}
    </div>
  );
}

function Notice({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'warn';
}) {
  return (
    <div
      className={`rounded-xl border p-6 text-sm ${
        tone === 'warn'
          ? 'border-amber-500/20 bg-amber-500/5 text-amber-200/80'
          : 'border-zinc-800 bg-zinc-900/50 text-zinc-400'
      }`}
    >
      {children}
    </div>
  );
}

function Footer({ snapshot, shown }: { snapshot: DealsSnapshot; shown: number }) {
  return (
    <footer className="mt-10 pt-6 border-t border-zinc-800 text-xs text-zinc-600 space-y-2">
      <p className="flex items-center gap-1.5">
        <CheckCircle2 className="w-3 h-3" />
        Showing {shown} of {snapshot.deals.length} deals, capped at{' '}
        {snapshot.perCategoryLimit} per category. This is a curated ranking, not a
        mirror of {snapshot.source}.
      </p>
      <p>
        Listings belong to their original posters; follow a link to view the full ad. No
        seller identities, contact details or precise locations are published here.{' '}
        <a
          href="https://github.com/Nimaho71/sentinel-blocket/blob/main/ETHICS.md"
          className="underline hover:text-zinc-400"
        >
          Ethics &amp; data policy
        </a>
      </p>
    </footer>
  );
}
