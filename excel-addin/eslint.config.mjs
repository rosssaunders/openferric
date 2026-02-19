import js from "@eslint/js";
import globals from "globals";
import importPlugin from "eslint-plugin-import";
import nPlugin from "eslint-plugin-n";
import promisePlugin from "eslint-plugin-promise";
import securityPlugin from "eslint-plugin-security";
import sonarPlugin from "eslint-plugin-sonarjs";
import unicornPlugin from "eslint-plugin-unicorn";
import tseslint from "typescript-eslint";

const tsProject = new URL("./tsconfig.json", import.meta.url).pathname;

export default tseslint.config(
  {
    ignores: [
      ".certs/**",
      "pkg/**",
      "node_modules/**",
      "*.js",
      "*.d.ts",
      "*.map"
    ]
  },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      parserOptions: {
        project: tsProject,
        tsconfigRootDir: new URL(".", import.meta.url).pathname
      },
      globals: {
        ...globals.browser,
        ...globals.node
      }
    },
    plugins: {
      import: importPlugin,
      n: nPlugin,
      promise: promisePlugin,
      security: securityPlugin,
      sonarjs: sonarPlugin,
      unicorn: unicornPlugin
    },
    settings: {
      "import/resolver": {
        typescript: {
          project: "./tsconfig.json"
        }
      }
    },
    rules: {
      "@typescript-eslint/array-type": ["error", { "default": "array-simple" }],
      "@typescript-eslint/consistent-type-assertions": ["error", { "assertionStyle": "never" }],
      "@typescript-eslint/consistent-type-definitions": ["error", "interface"],
      "@typescript-eslint/consistent-type-imports": ["error", { "prefer": "type-imports" }],
      "@typescript-eslint/explicit-function-return-type": ["error", { "allowExpressions": true }],
      "@typescript-eslint/no-confusing-void-expression": ["error", { "ignoreArrowShorthand": true }],
      "@typescript-eslint/no-import-type-side-effects": "error",
      "@typescript-eslint/no-unnecessary-condition": ["error", { "allowConstantLoopConditions": false }],
      "@typescript-eslint/no-useless-empty-export": "error",
      "@typescript-eslint/prefer-nullish-coalescing": "error",
      "@typescript-eslint/prefer-optional-chain": "error",

      "import/consistent-type-specifier-style": ["error", "prefer-top-level"],
      "import/first": "error",
      "import/newline-after-import": "error",
      "import/no-absolute-path": "error",
      "import/no-duplicates": "error",
      "import/no-named-as-default-member": "error",
      "import/no-unresolved": "error",
      "import/order": ["error", { "newlines-between": "always", "alphabetize": { "order": "asc", "caseInsensitive": true } }],

      "n/no-missing-import": "off",
      "n/no-unsupported-features/es-syntax": "off",

      "promise/always-return": "error",
      "promise/catch-or-return": ["error", { "allowFinally": true }],
      "promise/no-nesting": "error",
      "promise/no-return-wrap": "error",
      "promise/param-names": "error",
      "promise/prefer-await-to-callbacks": "error",
      "promise/prefer-await-to-then": "error",
      "promise/valid-params": "error",

      "security/detect-object-injection": "off",

      "sonarjs/no-duplicate-string": ["error", { "threshold": 5 }],

      "unicorn/catch-error-name": ["error", { "name": "error" }],
      "unicorn/consistent-destructuring": "error",
      "unicorn/no-array-for-each": "error",
      "unicorn/no-await-expression-member": "error",
      "unicorn/no-empty-file": "error",
      "unicorn/no-new-array": "error",
      "unicorn/no-null": "off",
      "unicorn/no-useless-undefined": "error",
      "unicorn/prefer-array-find": "error",
      "unicorn/prefer-array-flat": "error",
      "unicorn/prefer-array-flat-map": "error",
      "unicorn/prefer-modern-dom-apis": "error",
      "unicorn/prefer-node-protocol": "error",
      "unicorn/prefer-number-properties": "error",
      "unicorn/prefer-string-replace-all": "error",
      "unicorn/prefer-string-slice": "error",
      "unicorn/prefer-type-error": "error"
    }
  }
);
