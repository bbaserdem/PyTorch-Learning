{
  description = "Multi-framework ML learning environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            allowAliases = true;
          };
        };

        # Common packages shared between shells
        commonPackages = with pkgs; [
          # Python tooling
          uv
          # Node.js for any JS tooling needs
          nodejs-slim
          pnpm
          # Version control
          git
        ];

        # Common shell hook for Python/uv setup
        commonShellHook = ''
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
      in
      {
        devShells = {
          # Default dev shell - lightweight, no CUDA
          default = pkgs.mkShell {
            packages = commonPackages;
            shellHook = commonShellHook;
          };

          # CUDA-enabled dev shell for GPU computing
          cuda = pkgs.mkShell {
            packages = commonPackages ++ (with pkgs; [
              # Git tooling
              pre-commit
              gitleaks
              commitlint
              ruff

              # Hugging Face
              python313Packages.huggingface-hub

              # CUDA support
              cudaPackages.cudatoolkit
              cudaPackages.cudnn
              cudaPackages.cuda_cudart
              cudaPackages.cutensor
              cudaPackages.libcublas
              cudaPackages.libcurand
              cudaPackages.libcusparse
              gcc13

              # OpenCV and computer vision
              opencv4

              # PyArrow dependencies
              arrow-cpp

              # FFmpeg for av package
              ffmpeg

              # PostgreSQL client libraries
              postgresql

              # Build tools and compilers
              pkg-config
              cmake
              ninja
              gcc

              # Additional system libraries
              zlib
              libjpeg
              libpng
              libtiff
              eigen

              # Essential GUI dependencies only (minimal set for PyQt5)
              glib
              glibc
              libGL
              libxcb
              xcbutilxrm
              libxkbcommon
              fontconfig
              freetype
              dbus

              # Complete XCB and X11 libraries needed by Qt5 XCB plugin
              xorg.libX11
              xorg.libXi
              xorg.libXrender
              xorg.libXext
              xorg.libXrandr
              xorg.libXfixes
              xorg.libXcursor
              xorg.libXcomposite
              xorg.libXdamage
              xorg.libXinerama
              xorg.libXau
              xorg.libXdmcp
              xorg.xcbutil
              xorg.xcbutilimage
              xorg.xcbutilkeysyms
              xorg.xcbutilrenderutil
              xorg.xcbutilwm

              tk
              tcl
            ]);

            shellHook = commonShellHook + ''
              # Set CC to GCC 13 to avoid the version mismatch error
              export PATH=${pkgs.gcc13}/bin:$PATH

              # Essential GUI environment variables only
              export TCL_LIBRARY=${pkgs.tcl}/lib/tcl8.6
              export TK_LIBRARY=${pkgs.tk}/lib/tk8.6
            '';

            # Cuda env variables
            CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            CUDA_ROOT = "${pkgs.cudaPackages.cudatoolkit}";

            # Set CC to GCC 13 to avoid the version mismatch error
            CC = "${pkgs.gcc13}/bin/gcc";
            CXX = "${pkgs.gcc13}/bin/g++";

            # Environment variables to help packages find system libraries
            GDAL_DATA = "${pkgs.gdal}/share/gdal";
            PROJ_LIB = "${pkgs.proj}/share/proj";
            GDAL_LIBRARY_PATH = "${pkgs.gdal}/lib";
            GEOS_LIBRARY_PATH = "${pkgs.geos}/lib";

            # OpenCV
            OpenCV_DIR = "${pkgs.opencv4}/lib/cmake/opencv4";
            OPENCV_INCLUDE_DIRS = "${pkgs.opencv4}/include/opencv4";

            # Arrow
            ARROW_HOME = "${pkgs.arrow-cpp}";

            # Library path including CUDA and essential GUI libraries
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
              "/run/opengl-driver"
              cudaPackages.cudatoolkit
              cudaPackages.cudnn
              cudaPackages.cutensor
              cudaPackages.libcublas
              cudaPackages.libcurand
              cudaPackages.libcusparse

              glib
              glibc
              libGL
              libxcb
              xcbutilxrm
              libxkbcommon
              fontconfig
              freetype
              dbus

              xorg.libX11
              xorg.libXi
              xorg.libXrender
              xorg.libXext
              xorg.libXrandr
              xorg.libXfixes
              xorg.libXcursor
              xorg.libXcomposite
              xorg.libXdamage
              xorg.libXinerama
              xorg.libXau
              xorg.libXdmcp
              xorg.xcbutil
              xorg.xcbutilimage
              xorg.xcbutilkeysyms
              xorg.xcbutilrenderutil
              xorg.xcbutilwm
              libxkbcommon

              tk
              tcl
            ]);

            # Set LIBRARY_PATH to help the linker find the CUDA static libraries
            LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
              cudaPackages.cudatoolkit
            ]);

            # PKG_CONFIG_PATH for all libraries
            PKG_CONFIG_PATH =
              "${pkgs.gdal}/lib/pkgconfig"
              + ":"
              + "${pkgs.opencv4}/lib/pkgconfig"
              + ":"
              + "${pkgs.arrow-cpp}/lib/pkgconfig"
              + ":"
              + "${pkgs.postgresql}/lib/pkgconfig";
          };
        };
      }
    );
}
