/**
 * The published data contract.
 *
 * This mirrors exactly what `pipeline/publish.py` emits — a redacted, capped
 * snapshot. Fields absent here are absent by design, not by oversight:
 * descriptions, seller identity, GPS coordinates and postal codes are dropped
 * during ingest and never reach the browser.
 *
 * The site is a static reader of this file. It makes no API calls and holds no
 * credentials.
 */

/** Which build a deal is being scored against. */
export type ProfileId = 'ai-workstation' | 'gaming' | 'home-server';

/**
 * Whether a listing suits a profile.
 *
 * `unverified` is a first-class outcome, not a hedge. Most Blocket ads state
 * the graphics card and nothing else, so "the listing never mentioned its RAM"
 * has to be distinguishable from "this machine has too little RAM". Folding the
 * two together would hide most of the market behind a filter that looked like
 * it was working.
 */
export type FitVerdict = 'fits' | 'fits-after-upgrade' | 'unverified' | 'unfit';

export interface ProfileFit {
  profileId: ProfileId;
  verdict: FitVerdict;
  /** Requirements satisfied, unsatisfied, and not determinable, in that order. */
  met: string[];
  unmet: string[];
  unknown: string[];
  /** Unmet requirements no purchase can fix afterwards — soldered VRAM, mainly. */
  blockedBy: string[];
  /** Cost of the parts needed to bring the machine up to the profile's floor. */
  upgradeCostSek: number;
  /** Asking price plus that upgrade cost — what the buyer actually spends. */
  effectivePriceSek: number;
  /** The profile's headline ratio, e.g. "VRAM cost" in kr/GB. Null for `gaming`. */
  metric: string | null;
  metricUnit: string | null;
  metricValue: number | null;
  /**
   * Set when the card's model name covers several memory configurations and the
   * listing did not say which. The smaller was assumed; the user is told so.
   */
  vramCaveat: string | null;
}

/** Profile metadata for the selector, including why the profile exists. */
export interface ProfileInfo {
  id: ProfileId;
  label: string;
  rationale: string;
  requirements: { label: string; why: string; upgradeable: boolean }[];
  metric: string | null;
  metricUnit: string | null;
}

/** One live listing filling one slot of a parts basket. */
export interface BasketItem {
  slot: string;
  name: string;
  priceSek: number;
  title: string;
  city: string | null;
  url: string;
}

/**
 * Build-versus-buy for one profile.
 *
 * `excludedSlots` is part of the contract rather than a UI footnote. A parts
 * total that silently omits motherboard, PSU and case is not comparable to a
 * whole computer, and it is cheaper by exactly the amount it left out — so any
 * renderer that shows `totalSek` without the exclusions is misrepresenting it.
 */
export interface BasketSnapshot {
  profileId: ProfileId;
  totalSek: number;
  /** Distinct listings, i.e. how many strangers must be dealt with. */
  sellerCount: number;
  isComplete: boolean;
  missingSlots: string[];
  excludedSlots: string[];
  items: BasketItem[];
  bestPrebuiltId: string | null;
  bestPrebuiltPriceSek: number | null;
  /** Prebuilt minus basket. Null when the basket is incomplete and so not comparable. */
  savingSek: number | null;
}

/** A qualitative caveat found in the listing text. */
export interface RiskFlag {
  /** Stable identifier, e.g. "untested", "no_receipt", "ex_mining". */
  id: string;
  /** Short human-readable label. */
  label: string;
  /**
   * `info` is reassurance ("kvitto finns"), `warn` a hedge ("ej testad"), and
   * `high` the seller stating the item does not work ("defekt", "för delar").
   * A machine carrying a `high` flag is never offered as `bestPrebuiltId`.
   */
  severity: 'info' | 'warn' | 'high';
}

/** One component the pipeline resolved out of a listing. */
export interface ComponentRef {
  type: 'CPU' | 'GPU' | 'RAM' | 'MOTHERBOARD' | 'PSU' | 'STORAGE' | 'CASE' | 'OTHER';
  /** Canonical part name, e.g. "RTX 3060 12GB". */
  name: string;
  /** Index median for this part, or null when we lack the samples to say. */
  medianSek: number | null;
  /** How many observations back that median. Low n means low confidence. */
  sampleCount: number;
}

/**
 * The deterministic half of the valuation.
 *
 * Deliberately separate from the AI's contribution: this is arithmetic over
 * observed prices and is fully auditable. Nothing here is model output.
 */
export interface MarketMath {
  /** Sum of component medians. Null when too few parts could be priced. */
  estimatedPartsSek: number | null;
  /** listingPrice - estimatedParts. Negative means below market. */
  deltaSek: number | null;
  /** Weakest sample count across the components — the honest confidence signal. */
  minSampleCount: number;
  components: ComponentRef[];
}

/**
 * The AI's contribution: risk, never price.
 *
 * Keeping the model out of the pricing path is what stops hallucinated numbers
 * from reaching the ranking.
 */
export interface AiAssessment {
  /** One-line qualitative verdict. */
  summary: string;
  flags: RiskFlag[];
  /** Multiplier applied to the math delta when ranking, roughly 0.7–1.1. */
  confidence: number;
}

export interface Deal {
  id: string;
  title: string;
  priceSek: number;
  /** Attribution link back to the original listing. */
  url: string;
  category: string;
  /** Coarse place name only — never coordinates. */
  city: string | null;
  /** ISO 8601. */
  listedAt: string | null;
  /** Seller-declared condition, when the marketplace captured it structurally. */
  condition: string | null;
  math: MarketMath;
  ai: AiAssessment | null;
  /**
   * Fit against every profile, not just the selected one.
   *
   * The deployed site is a static file with no backend to re-query, so
   * switching profile has to be a client-side re-sort. Publishing all three
   * verdicts costs a few hundred bytes per deal and makes the selector instant;
   * the alternative is three snapshots that can disagree with each other.
   */
  fits: Record<ProfileId, ProfileFit>;
}

export interface DealsSnapshot {
  /** ISO 8601 timestamp of the pipeline run that produced this file. */
  generatedAt: string;
  /** Marketplace the deals came from. */
  source: string;
  /** Cap applied per category, surfaced so the UI can be honest about truncation. */
  perCategoryLimit: number;
  deals: Deal[];
  /** Selector metadata, so profile definitions live in one place — the pipeline. */
  profiles: ProfileInfo[];
  /** Build-versus-buy per profile, computed at publish time against live listings. */
  baskets: Record<ProfileId, BasketSnapshot>;
}
