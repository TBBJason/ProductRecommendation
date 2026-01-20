view: product_recommendations {
  sql_table_name: gold.product_recommendations ;;

  dimension: product_id {
    type: string
    primary_key: yes
    sql: ${TABLE}.product_id ;;
  }

  dimension: category {
    type: string
    sql: ${TABLE}.category ;;
  }

  measure: total_impressions {
    type: count_distinct
    sql: ${TABLE}.impression_id ;;
  }

  measure: total_clicks {
    type: count_distinct
    sql: ${TABLE}.click_id ;;
    filters: [interaction_type: "click"]
  }

  measure: ctr {
    type: number
    sql: 1.0 * ${total_clicks} / NULLIF(${total_impressions}, 0) ;;
    value_format_name: percent_2
  }

  measure: avg_price {
    type: average
    sql: ${TABLE}.price ;;
    value_format_name: usd
  }
}