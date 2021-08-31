/**
 * Implements Human build process
 * Used to generate prod builds for releases or by dev server to generate on-the-fly debug builds
 */

const fs = require('fs');
const path = require('path');
const log = require('@vladmandic/pilogger');
const esbuild = require('esbuild');
const rimraf = require('rimraf');
const lint = require('./lint.js');

let busy = false;

const config = {
  build: {
    banner: { js: `
    /*
      Human VRM
      homepage: <https://github.com/vladmandic/human>
      author: <https://github.com/vladmandic>'
    */` },
    tsconfig: './tsconfig.json',
    logLevel: 'error',
    bundle: true,
    metafile: true,
    target: 'es2018',
  },
  debug: {
    minifyWhitespace: false,
    minifyIdentifiers: false,
    minifySyntax: false,
  },
  production: {
    minifyWhitespace: true,
    minifyIdentifiers: true,
    minifySyntax: true,
  },
  buildLog: 'build.log',
  changelog: '../CHANGELOG.md',
  lintLocations: ['server/', 'src/'],
  cleanLocations: ['dist/*'],
};

const targets = {
  browser: {
    esm: {
      platform: 'browser',
      format: 'esm',
      entryPoints: ['src/kernels.ts'],
      outfile: 'dist/kernels.esm.js',
      external: ['fs', 'buffer', 'util', 'os'],
      sourcemap: true,
    },
  },
};

async function getStats(json) {
  const stats = {};
  if (json && json.metafile?.inputs && json.metafile?.outputs) {
    for (const [key, val] of Object.entries(json.metafile.inputs)) {
      if (key.startsWith('node_modules')) {
        stats.modules = (stats.modules || 0) + 1;
        stats.moduleBytes = (stats.moduleBytes || 0) + val.bytes;
      } else {
        stats.imports = (stats.imports || 0) + 1;
        stats.importBytes = (stats.importBytes || 0) + val.bytes;
      }
    }
    const files = [];
    for (const [key, val] of Object.entries(json.metafile.outputs)) {
      if (!key.endsWith('.map')) {
        files.push(key);
        stats.outputBytes = (stats.outputBytes || 0) + val.bytes;
      }
    }
    stats.outputFiles = files.join(', ');
  }
  return stats;
}

// rebuild typings

// rebuild on file change
async function build(f, msg, dev = false) {
  if (busy) {
    log.state('Build: busy...');
    setTimeout(() => build(f, msg, dev), 500);
    return;
  }
  busy = true;
  log.info('Build: file', msg, f, 'type:', dev ? 'debug' : 'production', 'config:', dev ? config.debug : config.production);
  // common build options
  try {
    // rebuild all target groups and types
    for (const [targetGroupName, targetGroup] of Object.entries(targets)) {
      for (const [targetName, targetOptions] of Object.entries(targetGroup)) {
        // if triggered from watch mode, rebuild only browser bundle
        // if ((require.main !== module) && ((targetGroupName === 'browserNoBundle') || (targetGroupName === 'nodeGPU'))) continue;
        const opt = dev ? config.debug : config.production;
        // @ts-ignore // eslint-typescript complains about string enums used in js code
        const meta = await esbuild.build({ ...config.build, ...opt, ...targetOptions });
        const stats = await getStats(meta);
        log.state(` target: ${targetGroupName} type: ${targetName}:`, stats);
      }
    }
  } catch (err) {
    // catch errors and print where it occured
    log.error('Build error', JSON.stringify(err.errors || err, null, 2));
    if (require.main === module) process.exit(1);
  }
  if (!dev) { // only for prod builds, skipped for dev build
    await lint.run(config.lintLocations); // run linter
  }
  if (require.main === module) process.exit(0);
  busy = false;
}

function clean() {
  log.info('Clean:', config.cleanLocations);
  for (const loc of config.cleanLocations) rimraf.sync(loc);
}

if (require.main === module) {
  config.buildLog = path.join(__dirname, config.buildLog);
  if (fs.existsSync(config.buildLog)) fs.unlinkSync(config.buildLog);
  log.logFile(config.buildLog);
  log.header();
  const toolchain = {
    esbuild: esbuild.version,
    eslint: lint.version,
  };
  log.info('Toolchain: ', toolchain);
  clean();
  build('all', 'startup');
} else {
  exports.build = build;
}
