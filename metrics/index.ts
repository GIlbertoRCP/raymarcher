import kerr from './kerr.st?raw';
import minkowski from './minkowski.st?raw';
import reissnerNordstrom from './reissner-nordstrom.st?raw';
import schwarzschild from './schwarzschild.st?raw';

/** The built-in spacetimes, keyed by file name. */
export const catalog = { schwarzschild, kerr, 'reissner-nordstrom': reissnerNordstrom, minkowski } as const;
export type CatalogName = keyof typeof catalog;
