{
  description = "PyTorch learning environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python tooling
            uv
            # Node.js for any JS tooling needs
            nodejs-slim
            pnpm
            # Version control
            git
          ];

          shellHook = ''
            # Setup node
            export PATH="./node_modules/.bin:$PATH"

            # Setup Python via uv
            export UV_PYTHON_PREFERENCE=only-managed
            
            # Create venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating Python virtual environment with uv..."
              uv venv
            fi

            # Activate the virtual environment
            source .venv/bin/activate

            # Sync dependencies if pyproject.toml exists
            if [ -f "pyproject.toml" ]; then
              echo "Syncing Python dependencies..."
              uv sync
            fi
          '';
        };
      }
    );
}
