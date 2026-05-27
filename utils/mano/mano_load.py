"""Local MANO model loading helpers for semantic_mano_ik CLI scripts."""
from __future__ import annotations

import os
from pathlib import Path

import smplx


def resolveManoPath(*, manoPath: str | None, projectRoot: str | Path) -> Path:
    if manoPath:
        resolved = Path(manoPath).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"MANO path does not exist: {resolved}")
    envPath = os.environ.get("MANO_PATH", "").strip()
    if envPath:
        resolved = Path(envPath).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"MANO_PATH does not exist: {resolved}")
    projectRoot = Path(projectRoot).resolve()
    localPath = projectRoot / "assets" / "mano"
    if localPath.exists():
        return localPath
    raise FileNotFoundError(
        "MANO model files were not found. Provide --mano-path, set MANO_PATH, "
        "or place the models under semantic_mano_ik/assets/mano."
    )


def createManoLayer(
    *,
    modelPath: str | Path,
    useFlatHandMean: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    modelPathObj = Path(modelPath)
    manoDir = modelPathObj / "mano"
    useSmplxCreate = manoDir.is_dir()
    modelDir = None
    if not useSmplxCreate:
        if (modelPathObj / "MANO_LEFT.pkl").is_file() and (modelPathObj / "MANO_RIGHT.pkl").is_file():
            modelDir = modelPathObj
        elif (modelPathObj / "models" / "MANO_LEFT.pkl").is_file() and (modelPathObj / "models" / "MANO_RIGHT.pkl").is_file():
            modelDir = modelPathObj / "models"
        else:
            raise FileNotFoundError(f"MANO model files were not found: {modelPathObj}")

    if useSmplxCreate:
        return {
            "right": smplx.create(
                str(modelPathObj), "mano",
                use_pca=False, is_rhand=True,
                num_pca_comps=45, is_Euler=False,
                flat_hand_mean=useFlatHandMean,
                scale=1.0,
            ).to(device),
            "left": smplx.create(
                str(modelPathObj), "mano",
                use_pca=False, is_rhand=False,
                num_pca_comps=45, is_Euler=False,
                flat_hand_mean=useFlatHandMean,
                scale=1.0,
            ).to(device),
        }
    return {
        "right": smplx.MANO(
            str(modelDir / "MANO_RIGHT.pkl"),
            is_rhand=True,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=useFlatHandMean,
        ).to(device),
        "left": smplx.MANO(
            str(modelDir / "MANO_LEFT.pkl"),
            is_rhand=False,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=useFlatHandMean,
        ).to(device),
    }
