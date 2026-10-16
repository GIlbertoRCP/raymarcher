import { catalog, type CatalogName } from '../../metrics/index.ts';

export type Background = 'stars' | 'grid' | 'none';

/** A catalog spacetime, or 'custom' for a metric typed into the worksheet. */
export type SpacetimeId = CatalogName | 'custom';

export interface Settings {
  spacetime: SpacetimeId;
  /** Parameter values for each spacetime, in the order the metric file declares them. */
  params: Partial<Record<SpacetimeId, number[]>>;
  /** Source of the custom metric, when spacetime is 'custom'. */
  customSource?: string;

  renderScale: number; // 0.25 to 1.0 internal resolution
  maxSteps: number;
  stepFraction: number; // RK4 step as a fraction of distance from the origin
  disk: boolean;
  background: Background;
  redshift: boolean;
  exposure: number;
  diskTemp: number; // peak disk temperature in kelvin

  camRadius: number; // in units of the metric's length scale (usually M)
  camYaw: number; // radians
  camPitch: number; // radians
  fovDegrees: number; // vertical field of view
  /** Colour pixels by ray fate (escaped / captured / disk / unfinished) instead of shading them. */
  debugFate?: boolean;
}

export function defaultSettings(): Settings {
  return {
    spacetime: 'kerr',
    params: {},
    renderScale: 0.75,
    maxSteps: 300,
    stepFraction: 0.03,
    disk: true,
    background: 'stars',
    redshift: false,
    exposure: 0.55,
    diskTemp: 4500,
    camRadius: 26,
    camYaw: 0.0,
    camPitch: 0.1,
    fovDegrees: 45,
  };
}

export const SPACETIMES = Object.keys(catalog) as CatalogName[];
