{
  description = "GPU Benchmark";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShell = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            # C++ development
            gcc12
            gdb
            cmake
            
            # CUDA toolkit
            cudatoolkit

            # Git
            gitSVN

            # Misc
            strace

          ];
          # shellHook = ''
          #   export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          # '';
        };
      }
    );
}
