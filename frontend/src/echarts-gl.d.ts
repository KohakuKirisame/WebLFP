declare module "echarts-gl/charts" {
  import type { EChartsExtensionInstaller } from "echarts/core";

  export const Scatter3DChart: EChartsExtensionInstaller;
}

declare module "echarts-gl/components" {
  import type { EChartsExtensionInstaller } from "echarts/core";

  export const Grid3DComponent: EChartsExtensionInstaller;
}
