import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const Guide: ThemeNote = defineNoteConfig({
    dir: 'guide',
    link: '/guide/',
    sidebar: [
        {
            text: 'Basic Info',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'basicinfo',
            items: [
                'intro',
                'framework',
                'install',
            ],
        },
        {
            text: 'Dataflex Selector',
            collapsed: false,
            icon: 'solar:filter-outline',
            prefix: 'selector',
            items: [
                'quickstart',
                'tutorial',
                'selector_less',
                'selector_offline_tsds',
                'selector_offline_near',
                'selector_zeroth'
            ],
        },
        {
            text: 'Dataflex Mixer',
            collapsed: false,
            icon: 'solar:shuffle-outline',
            prefix: 'mixer',
            items: [
                'quickstart',
                'tutorial',
                'doremi',
            ],
        },
        {
            text: 'Dataflex Weighter',
            collapsed: false,
            icon: 'solar:scale-outline',
            prefix: 'weighter',
            items: [
                'quickstart',
                'tutorial',
            ],
        },
    ],
})
