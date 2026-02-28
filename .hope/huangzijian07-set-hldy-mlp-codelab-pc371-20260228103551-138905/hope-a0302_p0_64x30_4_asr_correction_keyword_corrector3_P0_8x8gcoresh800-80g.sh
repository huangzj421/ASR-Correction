#!/usr/bin/env bash
export HOPE_BA_CLIENT_ID="speech"
export HOPE_BA_CLIENT_SECRET="oFHdnUYVfPdtL3rm2kMJp4kABPPDbdcA"
# HOPE_BA_SERVICE 无需修改
export HOPE_BA_SERVICE="AifreeHopeBa"
# 以下为必填项，如user不合法将以项目组负责人提交
export HOPE_BA_USER=huangzijian07
rm -rf .hopemeta
hope init ml-easy-job a0302_p0_64x30_4_asr_correction_keyword_corrector3_P0_8x8gcoresh800-80g --creator huangzijian07 || exit 1;
mv a0302_p0_64x30_4_asr_correction_keyword_corrector3_P0_8x8gcoresh800-80g-tmp.hope a0302_p0_64x30_4_asr_correction_keyword_corrector3_P0_8x8gcoresh800-80g.hope
hope run a0302_p0_64x30_4_asr_correction_keyword_corrector3_P0_8x8gcoresh800-80g.hope -Dmlp.sche.priority=P0 -Dhope.resource.experiment=a0112_p0_32x14_7