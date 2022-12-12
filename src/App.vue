<script lang="ts" setup>

import { ref, reactive, computed, onMounted } from 'vue'
import { computedAsync } from '@vueuse/core';

import * as ort from 'onnxruntime-web'

const form = reactive({
  'ThroatPain': 0,
  'Dyspnea': 0,
  'Fever': 0,
  'Cough': 0,
  'Headache': 0,
  'TasteDisorders': 0,
  'OlfactoryDisorders': 0,
  'Coryza': 0,
})

const loading = ref(true)

var sess: ort.InferenceSession

onMounted(async () => {
  sess = await ort.InferenceSession.create(
    await (
      await fetch('/fold_0.onnx')
    ).arrayBuffer())
  loading.value = false
})

const PositiveProba = computedAsync(async () => {
  if (!loading.value) {
    const keys = [
      'ThroatPain',
      'Dyspnea',
      'Fever',
      'Cough',
      'Headache',
      'TasteDisorders',
      'OlfactoryDisorders',
      'Coryza',
    ]

    const data = Float32Array.from(keys.map(k => (form as any)[k]))
    const tensor = new ort.Tensor('float32', data, [1, keys.length])

    const rst = await sess.run({
      input: tensor
    })

    console.log(rst)

    return (rst.probabilities.data[1] as number).toFixed(3)
  }

  return 0.0.toFixed(3)
}, 0.0.toFixed(3))

</script>

<template>
  <div class="container">
    <h2>新冠症状判别</h2>
    <div>根据症状判断新冠阳性概率</div>
    <el-form :model="form" label-width="120px" label-position="left">
      <el-form-item label="咽喉痛">
        <el-switch v-model="form.ThroatPain"></el-switch>
      </el-form-item>

      <el-form-item label="呼吸困难">
        <el-switch v-model="form.Dyspnea"></el-switch>
      </el-form-item>

      <el-form-item label="发热">
        <el-switch v-model="form.Fever"></el-switch>
      </el-form-item>

      <el-form-item label="咳嗽">
        <el-switch v-model="form.Cough"></el-switch>
      </el-form-item>

      <el-form-item label="头痛">
        <el-switch v-model="form.Headache"></el-switch>
      </el-form-item>

      <el-form-item label="味觉失灵">
        <el-switch v-model="form.TasteDisorders"></el-switch>
      </el-form-item>

      <el-form-item label="嗅觉失灵">
        <el-switch v-model="form.OlfactoryDisorders"></el-switch>
      </el-form-item>

      <el-form-item label="鼻炎">
        <el-switch v-model="form.Coryza"></el-switch>
      </el-form-item>

      <div v-if="loading">
        加载中...
      </div>
      <div v-else>
        <el-form-item label="阳性概率：">
          <div>
            {{ PositiveProba }}
          </div>
        </el-form-item>
      </div>

    </el-form>

    <small>
      本模型基于巴西某地的新冠病例健康数据构建，结果仅供参考。
      <p></p>
      [1] Viana dos Santos Santana, Íris ; C. M. da Silveira,, Andressa; Sobrinho, Alvaro; Chaves e Silva, Lenardo ; Dias da Silva, Leandro ; Freire de Souza Santos, Danilo ; Candeia, Edmar ; Perkusich, Angelo (2021), “A Brazilian dataset of symptomatic patients for screening the risk of COVID-19”, Mendeley Data, V5, doi: 10.17632/b7zcgmmwx4.5
    </small>
  </div>

</template>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
}

</style>
