# external/ — clone the three audited repositories here

None of the audited data or models is redistributed. The scripts read an unmodified clone of each
repository at the commit audited:

| directory | repository | commit |
|---|---|---|
| `external/RNAlight` | https://github.com/YangLab/RNAlight | `2b763b13778adb2ed2f0d734220db2237c487a3f` |
| `external/DVMnet` | https://github.com/liuliwei1980/DVMnet | `27bd6ea0d8782dc4b902bf79de9c5e8a38900e5f` |
| `external/LncPTPred` | https://github.com/zglabDIB/lncptpred | `621391287402cad35a8dadc1659fdd632148d24f` |

```bash
git clone https://github.com/YangLab/RNAlight external/RNAlight && git -C external/RNAlight checkout 2b763b13778adb2ed2f0d734220db2237c487a3f
git clone https://github.com/liuliwei1980/DVMnet external/DVMnet && git -C external/DVMnet checkout 27bd6ea0d8782dc4b902bf79de9c5e8a38900e5f
git clone https://github.com/zglabDIB/lncptpred external/LncPTPred && git -C external/LncPTPred checkout 621391287402cad35a8dadc1659fdd632148d24f
```

The commit metadata recorded at the time of the audit is in `revision_r2/provenance/`.
